use nokhwa::{
    pixel_format::{RgbAFormat, RgbFormat},
    utils::{CameraIndex, RequestedFormat, RequestedFormatType},
    Camera,
};
use piston_window::{graphics::clear, RenderEvent, Texture, TextureContext};
use piston_window::{PistonWindow, TextureSettings, WindowSettings};
use rust_faces::{BlazeFaceParams, FaceDetection, FaceDetectorBuilder, InferParams, Provider};

pub fn main() {
    let face_detector =
        FaceDetectorBuilder::new(FaceDetection::BlazeFace640(BlazeFaceParams::default()))
            .download()
            .infer_params(InferParams {
                provider: Provider::OrtCuda(0),
                inter_threads: Some(1),
                intra_threads: Some(8),
                ..Default::default()
            })
            .build()
            .expect("Fail to load the face detector.");

    let index = CameraIndex::Index(0);
    let requested =
        RequestedFormat::new::<RgbAFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
    let mut camera = Camera::new(index, requested).expect("Fail to open the camera.");

    let mut source;

    let mut window: PistonWindow = WindowSettings::new(
        "Face Detection",
        [camera.resolution().width(), camera.resolution().height()],
    )
    .vsync(true)
    .automatic_close(true)
    .exit_on_esc(true)
    .build()
    .unwrap();

    let mut texture_context = TextureContext {
        factory: window.factory.clone(),
        encoder: window.factory.create_command_buffer().into(),
    };

    let mut rgb_image;
    let mut rgba_image;

    let mut start = std::time::Instant::now();
    let mut frames = 0;
    loop {
        if let Some(event) = window.next() {
            if let Some(_) = event.render_args() {
                frames += 1;
            }
            source = camera.frame().expect("Fail to capture a frame.");

            rgb_image = source
                .decode_image::<RgbFormat>()
                .expect("Fail to decode the image.");
            rgba_image = source
                .decode_image::<RgbAFormat>()
                .expect("Fail to decode the image.");

            let detections = face_detector
                .detect(rgb_image)
                .expect("Fail to detect faces.");

            frames += 1;
            window.draw_2d(&event, |context, graphics, _| {
                clear([1.0; 4], graphics);
                let texture =
                    Texture::from_image(&mut texture_context, &rgba_image, &TextureSettings::new())
                        .expect("Fail to create a texture.");

                piston_window::image(&texture, context.transform, graphics);
            });

            for detection in detections {
                let x = detection.rect.x as f64;
                let y = detection.rect.y as f64;
                let width = detection.rect.width as f64;
                let height = detection.rect.height as f64;

                window.draw_2d(&event, |context, graphics, _| {
                    piston_window::line(
                        [1.0, 0.0, 0.0, 1.0],
                        1.0,
                        [x, y, x + width, y],
                        context.transform,
                        graphics,
                    );

                    piston_window::line(
                        [1.0, 0.0, 0.0, 1.0],
                        1.0,
                        [x, y, x, y + height],
                        context.transform,
                        graphics,
                    );

                    piston_window::line(
                        [1.0, 0.0, 0.0, 1.0],
                        1.0,
                        [x + width, y, x + width, y + height],
                        context.transform,
                        graphics,
                    );

                    piston_window::line(
                        [1.0, 0.0, 0.0, 1.0],
                        1.0,
                        [x, y + height, x + width, y + height],
                        context.transform,
                        graphics,
                    );
                });
            }
        } else {
            break;
        }

        if start.elapsed().as_secs() >= 1 {
            println!("FPS: {}", frames);
            frames = 0;
            start = std::time::Instant::now();
        }
    }
}
