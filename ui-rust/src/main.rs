use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use eframe::egui;
use serde::Deserialize;

const API_BASE: &str = "http://127.0.0.1:7860";

// ---------------------------------------------------------------------------
// API types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Default)]
struct HealthEntry {
    level: u32,
    active: u32,
    dead: u32,
    total: u32,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct TrainingStateResponse {
    step: u64,
    is_training: bool,
    health: Vec<HealthEntry>,
    latest_losses: HashMap<String, f64>,
    loss_history_len: usize,
}

// ---------------------------------------------------------------------------
// Shared state between polling thread and UI thread
// ---------------------------------------------------------------------------

#[derive(Default)]
struct SharedData {
    api_connected: bool,
    state: Option<TrainingStateResponse>,

    // Raw PNG bytes + version counters (version increments when new data arrives)
    codebook_png: Option<Vec<u8>>,
    codebook_version: u64,
    recon_png: Option<Vec<u8>>,
    recon_version: u64,
    loss_png: Option<Vec<u8>>,
    loss_version: u64,
    trajectory_l1_png: Option<Vec<u8>>,
    trajectory_l1_version: u64,
    trajectory_l2_png: Option<Vec<u8>>,
    trajectory_l2_version: u64,
    gradient_png: Option<Vec<u8>>,
    gradient_version: u64,
    decoded_png: Option<Vec<u8>>,
    decoded_version: u64,
}

// Config written by UI, read by polling thread
#[derive(Clone)]
struct PollingConfig {
    mode: String,
    main_interval_steps: u64,
    debug_interval_steps: u64,
}

impl Default for PollingConfig {
    fn default() -> Self {
        Self {
            mode: "Residuals".to_string(),
            main_interval_steps: 50,
            debug_interval_steps: 300,
        }
    }
}

// ---------------------------------------------------------------------------
// Cached texture: (handle, version_it_was_loaded_from)
// ---------------------------------------------------------------------------

type CachedTexture = Option<(egui::TextureHandle, u64)>;

// ---------------------------------------------------------------------------
// Main egui application
// ---------------------------------------------------------------------------

struct RqVaeApp {
    shared: Arc<Mutex<SharedData>>,
    config: Arc<Mutex<PollingConfig>>,

    // Cached textures (UI thread only)
    codebook_tex: CachedTexture,
    recon_tex: CachedTexture,
    loss_tex: CachedTexture,
    trajectory_l1_tex: CachedTexture,
    trajectory_l2_tex: CachedTexture,
    gradient_tex: CachedTexture,
    decoded_tex: CachedTexture,

    // Slider values (UI state)
    lambda_commit: f64,
    lambda_codebook: f64,
    lambda_wasserstein: f64,
    sinkhorn_epsilon: f64,
    main_interval: u64,
    debug_interval: u64,
    mode: String,

    // Last params sent to API (to debounce)
    last_sent: (f64, f64, f64, f64),

    // Blocking client for button clicks
    client: reqwest::blocking::Client,
}

impl RqVaeApp {
    fn new(
        _cc: &eframe::CreationContext<'_>,
        shared: Arc<Mutex<SharedData>>,
        config: Arc<Mutex<PollingConfig>>,
    ) -> Self {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .unwrap_or_default();

        Self {
            shared,
            config,
            codebook_tex: None,
            recon_tex: None,
            loss_tex: None,
            trajectory_l1_tex: None,
            trajectory_l2_tex: None,
            gradient_tex: None,
            decoded_tex: None,
            lambda_commit: 0.25,
            lambda_codebook: 1.0,
            lambda_wasserstein: 0.0,
            sinkhorn_epsilon: 0.2,
            main_interval: 50,
            debug_interval: 300,
            mode: "Residuals".to_string(),
            last_sent: (0.25, 1.0, 0.0, 0.2),
            client,
        }
    }

    /// Decode raw PNG bytes into an egui texture and cache it.
    fn maybe_update_texture(
        ctx: &egui::Context,
        cached: &mut CachedTexture,
        png: &Option<Vec<u8>>,
        version: u64,
        name: &str,
    ) {
        let Some(bytes) = png else { return };

        let cached_version = cached.as_ref().map(|(_, v)| *v).unwrap_or(u64::MAX);
        if cached_version == version {
            return; // already up to date
        }

        if let Ok(img) = image::load_from_memory(bytes) {
            let rgba = img.to_rgba8();
            let (w, h) = rgba.dimensions();
            let color_image = egui::ColorImage::from_rgba_unmultiplied(
                [w as usize, h as usize],
                rgba.as_raw(),
            );
            let handle =
                ctx.load_texture(name, color_image, egui::TextureOptions::LINEAR);
            *cached = Some((handle, version));
        }
    }

    /// Render a plot image or a placeholder box.
    fn show_plot(ui: &mut egui::Ui, cached: &CachedTexture, size: egui::Vec2) {
        if let Some((tex, _)) = cached {
            ui.add(egui::Image::new((tex.id(), size)));
        } else {
            let (rect, _) =
                ui.allocate_exact_size(size, egui::Sense::hover());
            ui.painter()
                .rect_filled(rect, 4.0, egui::Color32::from_gray(40));
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "loading…",
                egui::FontId::proportional(14.0),
                egui::Color32::GRAY,
            );
        }
    }

    fn post(&self, path: &str) {
        let url = format!("{}{}", API_BASE, path);
        let _ = self.client.post(&url).send();
    }

    fn send_params(&self) {
        let url = format!(
            "{}/params?lambda_commit={}&lambda_codebook={}&lambda_wasserstein={}&sinkhorn_epsilon={}",
            API_BASE,
            self.lambda_commit,
            self.lambda_codebook,
            self.lambda_wasserstein,
            self.sinkhorn_epsilon,
        );
        let _ = self.client.post(&url).send();
    }

    fn update_polling_config(&self) {
        if let Ok(mut cfg) = self.config.lock() {
            cfg.mode = self.mode.clone();
            cfg.main_interval_steps = self.main_interval;
            cfg.debug_interval_steps = self.debug_interval;
        }
    }
}

impl eframe::App for RqVaeApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Repaint continuously (polling thread also wakes us via request_repaint)
        ctx.request_repaint_after(Duration::from_millis(500));

        // Push updated slider config to polling thread
        self.update_polling_config();

        // Send params to API when sliders change
        let current = (
            self.lambda_commit,
            self.lambda_codebook,
            self.lambda_wasserstein,
            self.sinkhorn_epsilon,
        );
        if current != self.last_sent {
            self.send_params();
            self.last_sent = current;
        }

        // Snapshot shared data and update textures
        let (step, is_training, health, api_connected) = {
            let shared = self.shared.lock().unwrap();

            Self::maybe_update_texture(
                ctx,
                &mut self.codebook_tex,
                &shared.codebook_png,
                shared.codebook_version,
                "codebook",
            );
            Self::maybe_update_texture(
                ctx,
                &mut self.recon_tex,
                &shared.recon_png,
                shared.recon_version,
                "recon",
            );
            Self::maybe_update_texture(
                ctx,
                &mut self.loss_tex,
                &shared.loss_png,
                shared.loss_version,
                "loss",
            );
            Self::maybe_update_texture(
                ctx,
                &mut self.trajectory_l1_tex,
                &shared.trajectory_l1_png,
                shared.trajectory_l1_version,
                "traj_l1",
            );
            Self::maybe_update_texture(
                ctx,
                &mut self.trajectory_l2_tex,
                &shared.trajectory_l2_png,
                shared.trajectory_l2_version,
                "traj_l2",
            );
            Self::maybe_update_texture(
                ctx,
                &mut self.gradient_tex,
                &shared.gradient_png,
                shared.gradient_version,
                "gradient",
            );
            Self::maybe_update_texture(
                ctx,
                &mut self.decoded_tex,
                &shared.decoded_png,
                shared.decoded_version,
                "decoded",
            );

            let (step, is_training, health) = shared
                .state
                .as_ref()
                .map(|s| (s.step, s.is_training, s.health.clone()))
                .unwrap_or_default();
            (step, is_training, health, shared.api_connected)
        };

        // -----------------------------------------------------------------------
        // Render UI
        // -----------------------------------------------------------------------
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("RQ-VAE Explorer");
                ui.label(
                    "Interactive training visualization for Residual Quantized VAE on MNIST",
                );

                if !api_connected {
                    ui.colored_label(
                        egui::Color32::YELLOW,
                        "⚠ Connecting to Python API on http://127.0.0.1:7860 …",
                    );
                }

                ui.separator();

                // --- Training controls ---
                ui.horizontal(|ui| {
                    if ui.button("▶ Start Training").clicked() {
                        self.post("/training/start");
                    }
                    if ui.button("⏹ Stop").clicked() {
                        self.post("/training/stop");
                    }
                    if ui.button("↺ Reset").clicked() {
                        self.post("/training/reset");
                    }

                    let status = if is_training {
                        egui::RichText::new("🟢 Training").color(egui::Color32::GREEN)
                    } else {
                        egui::RichText::new("⏸ Paused").color(egui::Color32::GRAY)
                    };
                    ui.label(format!("Step: {:>6} |", step));
                    ui.label(status);
                });

                // --- Codebook health ---
                if !health.is_empty() {
                    ui.horizontal(|ui| {
                        ui.label("Codebook health:");
                        for h in &health {
                            let (symbol, color) = if h.dead == 0 {
                                ("✓", egui::Color32::GREEN)
                            } else {
                                ("⚠", egui::Color32::YELLOW)
                            };
                            ui.colored_label(
                                color,
                                format!("{} L{}: {}/{}", symbol, h.level, h.active, h.total),
                            );
                        }
                    });
                }

                ui.separator();

                // --- Main plot area ---
                let available = ui.available_width();
                let half = (available / 2.0 - 8.0).max(200.0);

                // Mode selector above the codebook plot
                ui.horizontal(|ui| {
                    ui.label("Plot 2 Mode:");
                    egui::ComboBox::from_id_salt("mode_select")
                        .selected_text(&self.mode)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut self.mode,
                                "Residuals".to_string(),
                                "Residuals",
                            );
                            ui.selectable_value(
                                &mut self.mode,
                                "Cumulative".to_string(),
                                "Cumulative",
                            );
                        });
                });

                ui.horizontal(|ui| {
                    // Left column: codebook plot
                    ui.vertical(|ui| {
                        ui.set_width(half);
                        ui.label(
                            egui::RichText::new("2D Codebook Visualization").strong(),
                        );
                        Self::show_plot(ui, &self.codebook_tex, egui::Vec2::new(half, 280.0));
                    });

                    ui.add_space(8.0);

                    // Right column: reconstructions + loss
                    ui.vertical(|ui| {
                        ui.set_width(half);
                        ui.label(
                            egui::RichText::new("Sample Reconstructions").strong(),
                        );
                        Self::show_plot(ui, &self.recon_tex, egui::Vec2::new(half, 130.0));
                        ui.add_space(4.0);
                        ui.label(egui::RichText::new("Loss Curves").strong());
                        Self::show_plot(ui, &self.loss_tex, egui::Vec2::new(half, 140.0));
                    });
                });

                ui.separator();

                // --- Parameter controls ---
                ui.label(egui::RichText::new("Training Parameters").strong());

                egui::Grid::new("params_grid")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("λ_commit (commitment loss weight):");
                        ui.add(
                            egui::Slider::new(&mut self.lambda_commit, 0.0..=2.0)
                                .step_by(0.05),
                        );
                        ui.end_row();

                        ui.label("λ_codebook (codebook loss weight):");
                        ui.add(
                            egui::Slider::new(&mut self.lambda_codebook, 0.0..=2.0)
                                .step_by(0.05),
                        );
                        ui.end_row();

                        ui.label("λ_wasserstein (optimal transport weight):");
                        ui.add(
                            egui::Slider::new(&mut self.lambda_wasserstein, 0.0..=1.0)
                                .step_by(0.01),
                        );
                        ui.end_row();

                        ui.label("Sinkhorn ε (transport softness):");
                        ui.add(
                            egui::Slider::new(&mut self.sinkhorn_epsilon, 0.01..=1.0)
                                .step_by(0.01),
                        );
                        ui.end_row();

                        ui.label("Main plot refresh (steps):");
                        ui.add(
                            egui::Slider::new(&mut self.main_interval, 10..=500)
                                .step_by(10.0),
                        );
                        ui.end_row();
                    });

                ui.separator();

                // --- Debug section ---
                egui::CollapsingHeader::new("Debug: Codebook Dynamics")
                    .default_open(false)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Debug refresh (steps):");
                            ui.add(
                                egui::Slider::new(&mut self.debug_interval, 50..=1000)
                                    .step_by(50.0),
                            );
                        });

                        ui.add_space(4.0);

                        ui.horizontal(|ui| {
                            ui.vertical(|ui| {
                                ui.set_width(half);
                                ui.label(
                                    egui::RichText::new("L1 Codebook Trajectory").strong(),
                                );
                                Self::show_plot(
                                    ui,
                                    &self.trajectory_l1_tex,
                                    egui::Vec2::new(half, 240.0),
                                );
                            });
                            ui.add_space(8.0);
                            ui.vertical(|ui| {
                                ui.set_width(half);
                                ui.label(
                                    egui::RichText::new("L2 Codebook Trajectory").strong(),
                                );
                                Self::show_plot(
                                    ui,
                                    &self.trajectory_l2_tex,
                                    egui::Vec2::new(half, 240.0),
                                );
                            });
                        });

                        ui.add_space(4.0);

                        ui.horizontal(|ui| {
                            ui.vertical(|ui| {
                                ui.set_width(half);
                                ui.label(
                                    egui::RichText::new("Gradient Magnitudes").strong(),
                                );
                                Self::show_plot(
                                    ui,
                                    &self.gradient_tex,
                                    egui::Vec2::new(half, 200.0),
                                );
                            });
                            ui.add_space(8.0);
                            ui.vertical(|ui| {
                                ui.set_width(half);
                                ui.label(
                                    egui::RichText::new("Decoded Codebook Combinations")
                                        .strong(),
                                );
                                Self::show_plot(
                                    ui,
                                    &self.decoded_tex,
                                    egui::Vec2::new(half, 200.0),
                                );
                            });
                        });
                    });
            });
        });
    }
}

// ---------------------------------------------------------------------------
// Polling thread
// ---------------------------------------------------------------------------

fn spawn_polling_thread(
    shared: Arc<Mutex<SharedData>>,
    config: Arc<Mutex<PollingConfig>>,
    ctx: egui::Context,
) {
    thread::spawn(move || {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .unwrap_or_default();

        let mut last_main_step: u64 = u64::MAX;
        let mut last_debug_step: u64 = u64::MAX;

        loop {
            // Read config
            let (mode, main_interval, debug_interval) = {
                let cfg = config.lock().unwrap();
                (
                    cfg.mode.clone(),
                    cfg.main_interval_steps,
                    cfg.debug_interval_steps,
                )
            };

            // Fetch training state
            let state_result = client
                .get(format!("{}/training/state", API_BASE))
                .send()
                .and_then(|r| r.json::<TrainingStateResponse>());

            match state_result {
                Ok(state) => {
                    let current_step = state.step;
                    {
                        let mut shared = shared.lock().unwrap();
                        shared.api_connected = true;
                        shared.state = Some(state);
                    }

                    // Main plots
                    let main_due = last_main_step == u64::MAX
                        || current_step.saturating_sub(last_main_step) >= main_interval;

                    if main_due {
                        fetch_png(
                            &client,
                            &format!("{}/plots/codebook?mode={}", API_BASE, mode),
                            |bytes, shared: &mut SharedData| {
                                shared.codebook_png = Some(bytes);
                                shared.codebook_version += 1;
                            },
                            &shared,
                        );
                        fetch_png(
                            &client,
                            &format!("{}/plots/reconstructions", API_BASE),
                            |bytes, shared: &mut SharedData| {
                                shared.recon_png = Some(bytes);
                                shared.recon_version += 1;
                            },
                            &shared,
                        );
                        fetch_png(
                            &client,
                            &format!("{}/plots/loss", API_BASE),
                            |bytes, shared: &mut SharedData| {
                                shared.loss_png = Some(bytes);
                                shared.loss_version += 1;
                            },
                            &shared,
                        );
                        last_main_step = current_step;
                    }

                    // Debug plots
                    let debug_due = last_debug_step == u64::MAX
                        || current_step.saturating_sub(last_debug_step) >= debug_interval;

                    if debug_due {
                        fetch_png(
                            &client,
                            &format!("{}/plots/trajectory?level=0", API_BASE),
                            |bytes, shared: &mut SharedData| {
                                shared.trajectory_l1_png = Some(bytes);
                                shared.trajectory_l1_version += 1;
                            },
                            &shared,
                        );
                        fetch_png(
                            &client,
                            &format!("{}/plots/trajectory?level=1", API_BASE),
                            |bytes, shared: &mut SharedData| {
                                shared.trajectory_l2_png = Some(bytes);
                                shared.trajectory_l2_version += 1;
                            },
                            &shared,
                        );
                        fetch_png(
                            &client,
                            &format!("{}/plots/gradients", API_BASE),
                            |bytes, shared: &mut SharedData| {
                                shared.gradient_png = Some(bytes);
                                shared.gradient_version += 1;
                            },
                            &shared,
                        );
                        fetch_png(
                            &client,
                            &format!("{}/plots/decoded", API_BASE),
                            |bytes, shared: &mut SharedData| {
                                shared.decoded_png = Some(bytes);
                                shared.decoded_version += 1;
                            },
                            &shared,
                        );
                        last_debug_step = current_step;
                    }
                }
                Err(_) => {
                    shared.lock().unwrap().api_connected = false;
                }
            }

            ctx.request_repaint();
            thread::sleep(Duration::from_millis(500));
        }
    });
}

fn fetch_png<F>(
    client: &reqwest::blocking::Client,
    url: &str,
    mut update: F,
    shared: &Arc<Mutex<SharedData>>,
) where
    F: FnMut(Vec<u8>, &mut SharedData),
{
    if let Ok(resp) = client.get(url).send() {
        if let Ok(bytes) = resp.bytes() {
            let mut shared = shared.lock().unwrap();
            update(bytes.to_vec(), &mut shared);
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() -> eframe::Result {
    let shared = Arc::new(Mutex::new(SharedData::default()));
    let config = Arc::new(Mutex::new(PollingConfig::default()));

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 900.0])
            .with_title("RQ-VAE Explorer"),
        ..Default::default()
    };

    let shared_clone = Arc::clone(&shared);
    let config_clone = Arc::clone(&config);

    eframe::run_native(
        "RQ-VAE Explorer",
        options,
        Box::new(move |cc| {
            // Start polling thread with egui context for request_repaint
            spawn_polling_thread(
                Arc::clone(&shared_clone),
                Arc::clone(&config_clone),
                cc.egui_ctx.clone(),
            );

            Ok(Box::new(RqVaeApp::new(cc, shared_clone, config_clone)))
        }),
    )
}
