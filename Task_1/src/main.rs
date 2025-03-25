use clap::Parser;
use opencv::{
    core,
    imgcodecs,
    imgproc,
    prelude::*,
    types::VectorOfi32,
};
use std::fs;
use std::path::{Path, PathBuf};
use tch::{CModule, Device, Kind, Tensor};
use std::error::Error;

/// Command line arguments.
#[derive(Parser)]
struct Args {
    /// Input directory containing images.
    #[clap(short, long)]
    input_dir: String,
    /// Output directory for cropped faces.
    #[clap(short, long)]
    output_dir: String,
    /// Path to the TorchScript model file (model.pt).
    #[clap(short, long, default_value = "model.pt")]
    model_path: String,
    /// Confidence threshold for face detection.
    #[clap(short, long, default_value = "0.5")]
    conf_thresh: f32,
}

/// Center-crop an image around a bounding box with a scaling factor.
fn center_crop(img: &Mat, bbox: core::Rect, scale: f32) -> opencv::Result<Mat> {
    let cx = bbox.x + bbox.width / 2;
    let cy = bbox.y + bbox.height / 2;
    let crop_size = ((if bbox.width > bbox.height { bbox.width } else { bbox.height } as f32) * scale) as i32;
    
    // Ensure crop_size is positive
    if crop_size <= 0 {
        return Err(opencv::Error::new(
            opencv::core::StsError as i32,
            "Invalid crop size".to_string(),
        ));
    }

    // Calculate boundaries with bounds checking
    let sx = (cx - crop_size / 2).clamp(0, img.cols());
    let sy = (cy - crop_size / 2).clamp(0, img.rows());
    let ex = (cx + crop_size / 2).clamp(0, img.cols());
    let ey = (cy + crop_size / 2).clamp(0, img.rows());
    
    // Ensure we have valid dimensions
    let width = ex - sx;
    let height = ey - sy;
    if width <= 0 || height <= 0 {
        return Err(opencv::Error::new(
            opencv::core::StsError as i32,
            "Invalid crop dimensions".to_string(),
        ));
    }

    let rect = core::Rect::new(sx, sy, width, height);
    Mat::roi(img, rect)
}

/// Process a single image: load, run inference, and crop if exactly one face is detected.
fn process_image(image_path: &Path, model: &CModule, conf_thresh: f32) -> opencv::Result<Option<Mat>> {
    // Load the image from disk.
    let img = imgcodecs::imread(image_path.to_str().unwrap(), imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        return Ok(None);
    }

    // Convert the image from BGR to RGB.
    let mut img_rgb = Mat::default();
    imgproc::cvt_color(&img, &mut img_rgb, imgproc::COLOR_BGR2RGB, 0)?;

    // Convert Mat data to a Tensor.
    let data = img_rgb.data_bytes()?; // Use `?` to get &[u8]
    let rows = img_rgb.rows();
    let cols = img_rgb.cols();
    let channels = img_rgb.channels(); // returns i32
    let tensor = Tensor::of_slice(data)
        .reshape(&[rows as i64, cols as i64, channels as i64])
        .to_device(Device::Cpu)
        .to_kind(Kind::Float) / 255.0;
    // Rearrange dimensions to [batch, channels, height, width].
    let input_tensor = tensor.permute(&[2, 0, 1]).unsqueeze(0);

    // Run inference using forward_ts and convert the output to a Tensor.
    let output = model.forward_ts(&[input_tensor]).expect("Model inference failed");
    let detections: Tensor = output.try_into().expect("Failed to convert model output");

    // Verify output dimensions; we expect [N, 6]: [x1, y1, x2, y2, confidence, class].
    let sizes = detections.size();
    if sizes.len() != 2 || sizes[1] < 6 {
        return Ok(None);
    }
    if sizes[0] == 0 {
        return Ok(None);
    }

    // Filter detections by confidence.
    let confs = detections.select(1, 4);
    let mask = confs.ge(conf_thresh as f64);
    let nonzero = mask.nonzero();
    let filtered = detections.index_select(0, &nonzero.squeeze());

    // Only process images with exactly one detection.
    if filtered.size()[0] != 1 {
        return Ok(None);
    }
    let detection = filtered.get(0);
    let x1 = detection.double_value(&[0]) as i32;
    let y1 = detection.double_value(&[1]) as i32;
    let x2 = detection.double_value(&[2]) as i32;
    let y2 = detection.double_value(&[3]) as i32;
    let width = (x2 - x1).abs();
    let height = (y2 - y1).abs();
    let bbox = core::Rect::new(x1, y1, width, height);

    // Crop the image using center_crop (scaling factor 2.0).
    let cropped = center_crop(&img, bbox, 2.0)?;
    Ok(Some(cropped))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    // Create the output directory if it doesn't exist.
    fs::create_dir_all(&args.output_dir)?;

    // Load the TorchScript model.
    let model = CModule::load(&args.model_path).expect("Failed to load model");

    // Process each image in the input directory.
    let input_path = Path::new(&args.input_dir);
    for entry in fs::read_dir(input_path)? {
        let entry = entry?;
        let path = entry.path();
        if let Some(ext) = path.extension() {
            if ext.eq_ignore_ascii_case("jpg")
                || ext.eq_ignore_ascii_case("jpeg")
                || ext.eq_ignore_ascii_case("png")
            {
                match process_image(&path, &model, args.conf_thresh) {
                    Ok(Some(cropped)) => {
                        let file_name = path.file_name().unwrap().to_str().unwrap();
                        let out_path = PathBuf::from(&args.output_dir).join(file_name);
                        // Save the cropped image.
                        imgcodecs::imwrite(out_path.to_str().unwrap(), &cropped, &VectorOfi32::new())?;
                        println!("Processed: {}", file_name);
                    }
                    Ok(None) => println!("Skipped (invalid/multiple faces): {:?}", path),
                    Err(e) => eprintln!("Error processing {:?}: {:?}", path, e),
                }
            }
        }
    }
    Ok(())
}
