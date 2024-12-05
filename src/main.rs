use ndarray::Array3;
use nokhwa::{
    pixel_format::RgbAFormat,
    utils::{CameraIndex, RequestedFormat, RequestedFormatType},
    Camera,
};
use rust_faces::{BlazeFaceParams, FaceDetection, FaceDetectorBuilder, InferParams, Provider}; 

pub fn main() {
    let face_detector =
        FaceDetectorBuilder::new(FaceDetection::BlazeFace640(BlazeFaceParams::default()))
            .download()
            .infer_params(InferParams {
                provider: Provider::OrtCuda(0),
                inter_threads: Some(1),
                intra_threads: Some(1),
                ..Default::default()
            })
            .build()
            .expect("Fail to load the face detector.");

    let index = CameraIndex::Index(0);
    let requested =
        RequestedFormat::new::<RgbAFormat>(RequestedFormatType::AbsoluteHighestResolution);
    let mut camera = Camera::new(index, requested).expect("Fail to open the camera.");

    let mut source = camera
        .frame()
        .expect("Fail to capture a frame.")
        .decode_image::<RgbAFormat>()
        .unwrap();

    let mut image_array = Array3::from_shape_fn(
        (source.height() as usize, source.width() as usize, 3),
        |(y, x, c)| source[(x as u32, y as u32)][c],
    );

    loop {
        source = camera
            .frame()
            .expect("Fail to capture a frame.")
            .decode_image::<RgbAFormat>()
            .unwrap();
        image_array.assign(&Array3::from_shape_fn(
            (source.height() as usize, source.width() as usize, 3),
            |(y, x, c)| source[(x as u32, y as u32)][c],
        ));

        let _ = face_detector
            .detect(image_array.view().into_dyn())
            .expect("Fail to detect faces.");
    }
}