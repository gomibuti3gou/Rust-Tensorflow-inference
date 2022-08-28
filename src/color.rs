use std::error::Error;
use std::path::Path;
use std::result::Result;
use tensorflow::Code;
use tensorflow::Graph;
use tensorflow::SavedModelBundle;
use tensorflow::SessionOptions;
use tensorflow::SessionRunArgs;
use tensorflow::Status;
use tensorflow::Tensor;
use tensorflow::DEFAULT_SERVING_SIGNATURE_DEF_KEY;

use image::io::Reader as ImageReader;
use image::GenericImageView;

pub fn cnn() -> Result<(),Box<dyn Error>> {
  let export_dir = "/home/color/mnist_savedmodel";
    if !Path::new(export_dir).exists() {
        return Err(Box::new(
            Status::new_set(
                Code::NotFound,
                &format!(
                    "Run 'python mnist_savedmodel.py' to generate \
                     {} and try again.",
                     export_dir
                ),
            ).unwrap(),
        ));
    }

    let mut x = Tensor::new(&[1,32,32,3]);
    ////home/color/mnist_savedmodel/sample.png
    let img = ImageReader::open("/home/color/mnist_savedmodel/sample.png")?.decode()?;
    for (i,(_,_,pixel)) in img.pixels().enumerate() {
      x[3 * i] = pixel.0[0] as f32;
      x[3 * i + 1] = pixel.0[1] as f32;
      x[3 * i + 2] = pixel.0[2] as f32;
    }
    
    println!("x?? {:?}",x);
    //println!("{:?}",img.pixels());

    let mut graph = Graph::new();
    let bundle = SavedModelBundle::load(&SessionOptions::new(),&["serve"],&mut graph, export_dir)?;
    let session = &bundle.session;

    let signature = bundle
        .meta_graph_def()
        .get_signature(DEFAULT_SERVING_SIGNATURE_DEF_KEY)?;
    let input_info = signature.get_input("input")?;
    let op_x = graph.operation_by_name_required(&input_info.name().name)?;
    let output_info = signature.get_output("sequential_3")?;
    let op_predict = graph.operation_by_name_required(&output_info.name().name)?;
    let mut args = SessionRunArgs::new();
    println!("Color type");
    //add_feed(&mut self,operation, index,tensor)
    //グラフに供給される入力を追加します。
    //インデックスは、フィードする操作の出力を選択　ここでは、０を指定
    args.add_feed(&op_x,0,&x);
    let output = args.request_fetch(&op_predict,0);
    session.run(&mut args)?;

    let mut res = Vec::new();
    let output: Tensor<f32> = args.fetch(output)?;
    for i in 0..10 {
        res.push(output[i]);
    }

    println!("{:?}", res);

    Ok(())
}