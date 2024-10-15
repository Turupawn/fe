mod backend;
mod cli_args;
mod functionality;
mod logger;
mod lsp_actor;
mod lsp_actor_service;
mod lsp_streams;
mod server;
mod util;

use std::backtrace::Backtrace;
use std::net::SocketAddr;

use async_compat::CompatExt;
use async_lsp::concurrency::ConcurrencyLayer;
use async_lsp::panic::CatchUnwindLayer;
use async_lsp::server::LifecycleLayer;
use async_lsp::tracing::TracingLayer;
use clap::Parser;
use cli_args::{CliArgs, Commands};
use futures::io::AsyncReadExt;
use futures::StreamExt;
use futures_net::TcpListener;
use server::setup;
use tracing::instrument::WithSubscriber;
use tracing::subscriber::set_default;
// use lsp_actor_service::LspActorService;
use tracing::{error, info};

use async_lsp::client_monitor::ClientProcessMonitorLayer;
use backend::db::Jar;
// use lsp_actor::{ActOnNotification, ActOnRequest};
use tower::ServiceBuilder;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::registry;
use tracing_tree::HierarchicalLayer;

#[tokio::main]
async fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");
    let std_tracing = registry()
        .with(tracing_subscriber::filter::LevelFilter::INFO)
        .with(
            HierarchicalLayer::new(2)
                .with_thread_ids(true)
                // .with_thread_names(true)
                // .with_indent_lines(true)
                // .with_bracketed_fields(true)
                .with_writer(std::io::stderr),
        );
    let logging = set_default(std_tracing);

    // Set up a panic hook
    std::panic::set_hook(Box::new(|panic_info| {
        // Extract the panic message
        let payload = panic_info.payload();
        let message = if let Some(s) = payload.downcast_ref::<&str>() {
            *s
        } else if let Some(s) = payload.downcast_ref::<String>() {
            &s[..]
        } else {
            "Unknown panic message"
        };

        // Get the location of the panic if available
        let location = if let Some(location) = panic_info.location() {
            format!(" at {}:{}", location.file(), location.line())
        } else {
            String::from("Unknown location")
        };

        // Capture the backtrace
        let backtrace = Backtrace::capture();

        // Log the panic information and backtrace
        tracing::error!(
            "Panic occurred{}: {}\nBacktrace:\n{:?}",
            location,
            message,
            backtrace
        );
    }));

    // Parse CLI arguments
    let args = CliArgs::parse();

    match args.command {
        Some(Commands::Tcp(tcp_args)) => {
            // Start server with TCP listener
            start_tcp_server(tcp_args.port).await;
        }
        None => {
            // Start server with stdio
            start_stdio_server().await;
        }
    }

    drop(logging);
}

async fn start_stdio_server() {
    let (server, _) = async_lsp::MainLoop::new_server(|client| {
        let tracing_layer = TracingLayer::default();
        let lsp_service = setup(client.clone(), "LSP actor".to_string());
        ServiceBuilder::new()
            .layer(LifecycleLayer::default())
            .layer(CatchUnwindLayer::default())
            .layer(ConcurrencyLayer::default())
            .layer(tracing_layer)
            .layer(ClientProcessMonitorLayer::new(client.clone()))
            .service(lsp_service)
    });

    let (stdin, stdout) = (tokio::io::stdin(), tokio::io::stdout());
    let (stdin, stdout) = (stdin.compat(), stdout.compat());

    match server.run_buffered(stdin, stdout).await {
        Ok(_) => info!("Server finished successfully"),
        Err(e) => error!("Server error: {:?}", e),
    }
}

async fn start_tcp_server(port: u16) {
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let mut listener = TcpListener::bind(&addr).expect("Failed to bind to address");
    let mut incoming = listener.incoming();

    info!("LSP server is listening on {}", addr);

    while let Some(Ok(stream)) = incoming.next().with_current_subscriber().await {
        let client_address = stream.peer_addr().unwrap();
        let tracing_layer = TracingLayer::default();
        let task = async move {
            let (server, _) = async_lsp::MainLoop::new_server(|client| {
                let router = setup(client.clone(), format!("LSP actor for {client_address}"));
                ServiceBuilder::new()
                    .layer(tracing_layer)
                    .layer(LifecycleLayer::default())
                    .layer(CatchUnwindLayer::default())
                    .layer(ConcurrencyLayer::default())
                    .layer(ClientProcessMonitorLayer::new(client.clone()))
                    .service(router)
            });

            let (read, write) = stream.split();
            if let Err(e) = server.run_buffered(read, write).await {
                error!("Server error for client {}: {:?}", client_address, e);
            } else {
                info!("Client {} disconnected", client_address);
            }
        };
        tokio::spawn(task.with_current_subscriber());
    }
}
