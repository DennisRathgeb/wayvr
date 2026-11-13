//! Screens straight from KWin (`zkde_screencast_unstable_v1`).
//!
//! KWin hands out one PipeWire node per output on request, which is what its own portal does
//! after the user has picked a screen in the dialog. Asking KWin directly needs no dialog and
//! cannot end up with the wrong screen. KWin offers the interface only to programs it trusts
//! (a desktop file that names the interface in `X-KDE-Wayland-Interfaces`); without that the
//! bind fails and the caller falls back to the portal.

use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use wayland_client::{
    Connection, Dispatch, Proxy, QueueHandle,
    globals::{GlobalListContents, registry_queue_init},
    protocol::{
        wl_output::{self, WlOutput},
        wl_registry::WlRegistry,
    },
};
use wayland_protocols_plasma::screencast::v1::client::{
    zkde_screencast_stream_unstable_v1::{self, ZkdeScreencastStreamUnstableV1},
    zkde_screencast_unstable_v1::ZkdeScreencastUnstableV1,
};

/// `zkde_screencast_unstable_v1.pointer`
const POINTER_EMBEDDED: u32 = 2;
const POINTER_METADATA: u32 = 4;

#[derive(Default)]
struct State {
    /// registry name of the output -> its name (`DP-5`)
    names: HashMap<u32, String>,
    /// registry name of the output -> PipeWire node, or why there is none
    nodes: HashMap<u32, Result<u32, String>>,
}

/// One PipeWire node per output, by output name. `None` when KWin does not offer the interface
/// to this program. The streams live as long as the process: the connection is handed to a
/// thread that keeps reading it.
pub fn stream_outputs(cursor_metadata: bool) -> Option<HashMap<String, u32>> {
    let connection = Connection::connect_to_env().ok()?;
    let (globals, mut queue) = registry_queue_init::<State>(&connection).ok()?;
    let qh = queue.handle();

    let screencast: ZkdeScreencastUnstableV1 = globals
        .bind(&qh, 1..=4, ())
        .inspect_err(|e| log::info!("KWin screencast not offered to this program: {e}"))
        .ok()?;

    let mut state = State::default();
    let mut outputs = Vec::new();
    for g in globals.contents().clone_list() {
        if g.interface == WlOutput::interface().name && g.version >= 4 {
            let output: WlOutput = globals.registry().bind(g.name, 4, &qh, g.name);
            outputs.push((g.name, output));
        }
    }
    // the outputs' names
    queue.roundtrip(&mut state).ok()?;

    let pointer = if cursor_metadata {
        POINTER_METADATA
    } else {
        POINTER_EMBEDDED
    };
    let mut streams = Vec::new();
    for (id, output) in &outputs {
        streams.push(screencast.stream_output(output, pointer, &qh, *id));
    }

    let deadline = Instant::now() + Duration::from_secs(5);
    while state.nodes.len() < outputs.len() && Instant::now() < deadline {
        if queue.roundtrip(&mut state).is_err() {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    let mut by_name = HashMap::new();
    for (id, _) in &outputs {
        let name = state.names.get(id).cloned().unwrap_or_default();
        match state.nodes.get(id) {
            Some(Ok(node)) => {
                log::info!("{name}: PipeWire node {node} from KWin");
                by_name.insert(name, *node);
            }
            Some(Err(e)) => log::warn!("{name}: KWin gave no stream: {e}"),
            None => log::warn!("{name}: KWin did not answer the stream request"),
        }
    }

    // Keep the connection, and with it the streams, for the life of the process.
    let _ = std::thread::Builder::new()
        .name("kde-screencast".into())
        .spawn(move || {
            let _keep = (connection, screencast, outputs, streams);
            while queue.blocking_dispatch(&mut state).is_ok() {}
            log::warn!("KWin screencast connection closed");
        });

    Some(by_name)
}

impl Dispatch<WlOutput, u32> for State {
    fn event(
        state: &mut Self,
        _proxy: &WlOutput,
        event: wl_output::Event,
        id: &u32,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        if let wl_output::Event::Name { name } = event {
            state.names.insert(*id, name);
        }
    }
}

impl Dispatch<ZkdeScreencastStreamUnstableV1, u32> for State {
    fn event(
        state: &mut Self,
        _proxy: &ZkdeScreencastStreamUnstableV1,
        event: zkde_screencast_stream_unstable_v1::Event,
        id: &u32,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        match event {
            zkde_screencast_stream_unstable_v1::Event::Created { node } => {
                state.nodes.insert(*id, Ok(node));
            }
            zkde_screencast_stream_unstable_v1::Event::Failed { error } => {
                state.nodes.insert(*id, Err(error));
            }
            zkde_screencast_stream_unstable_v1::Event::Closed => {
                let name = state.names.get(id).cloned().unwrap_or_default();
                log::warn!("{name}: KWin closed the screen stream");
            }
            _ => {}
        }
    }
}

impl Dispatch<ZkdeScreencastUnstableV1, ()> for State {
    fn event(
        _state: &mut Self,
        _proxy: &ZkdeScreencastUnstableV1,
        _event: <ZkdeScreencastUnstableV1 as Proxy>::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<WlRegistry, GlobalListContents> for State {
    fn event(
        _state: &mut Self,
        _proxy: &WlRegistry,
        _event: <WlRegistry as Proxy>::Event,
        _data: &GlobalListContents,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}
