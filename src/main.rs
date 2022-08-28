use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use std::path::Path;
use std::result::Result;
use tensorflow::Code;
use tensorflow::Graph;
use tensorflow::SavedModelBundle;
use tensorflow::Session;
use tensorflow::SessionOptions;
use tensorflow::SessionRunArgs;
use tensorflow::Status;
use tensorflow::Tensor;
use tensorflow::ImportGraphDefOptions;
use tensorflow::DEFAULT_SERVING_SIGNATURE_DEF_KEY;
use std::fs;
use std::io::Write;

use image::io::Reader as ImageReader;
use image::GenericImageView;
use image::imageops::FilterType;
//画像のサイズ
const HEIGHT:u32 = 28;
const WIDTH:u32 = 28;

mod module;
mod color;
mod path;

struct ImagePredict(String,usize);


fn model_load(model_dir: &str,data_dir : &str) -> Result<Vec<ImagePredict>,Box<dyn Error>> {
    ///home/examples/mnist_savedmodel/saved_model.pb
    let mut res:Vec<ImagePredict> = Vec::new();
    //let export_dir = "./examples/mnist_savedmodel";
    if !Path::new(model_dir).exists() {
        return Err(Box::new(
            Status::new_set(
                Code::NotFound,
                &format!(
                    "Run 'python mnist_savedmodel.py' to generate \
                     {} and try again.",
                     model_dir
                ),
            ).unwrap(),
        ));
    }

    let mut graph = Graph::new();
    let bundle = SavedModelBundle::load(&SessionOptions::new(),&["serve"],&mut graph, model_dir)?;
    let session = &bundle.session;

    let signature = bundle
        .meta_graph_def()
        .get_signature(DEFAULT_SERVING_SIGNATURE_DEF_KEY)?;
    let input_info = signature.get_input("input")?;
    let op_x = graph.operation_by_name_required(&input_info.name().name)?;
    //cnn_7 output
    let output_info = signature.get_output("output")?;
    let op_predict = graph.operation_by_name_required(&output_info.name().name)?;

    let mut x = Tensor::new(&[WIDTH.into(),HEIGHT.into()]);
    let files = path::getDirPath(data_dir)?;
    
    for file in files {
        //let img = ImageReader::open(&file)?.decode()?;
        let img = image::open(&file)?;
        let img = img.resize(WIDTH,HEIGHT,FilterType::Lanczos3);
        for (i,(_,_,pixel)) in img.pixels().enumerate() {
            x[i] = pixel.0[0] as f32 / 255.0f32;
        }
        let mut args = SessionRunArgs::new();
        //add_feed(&mut self,operation, index,tensor)
        //グラフに供給される入力を追加します。
        //インデックスは、フィードする操作の出力を選択　ここでは、０を指定
        args.add_feed(&op_x,0,&x);
        let output = args.request_fetch(&op_predict,0);
        session.run(&mut args)?;

        let mut res2 = Vec::new();
        let output: Tensor<f32> = args.fetch(output)?;
        let mut ch:f32 = 0.0;
        let mut flag:usize= 0;
        for i in 0..10 {
            res2.push(output[i]);
            if ch <= res2[i] {
                ch = res2[i];
                flag = i;
            }
        }
        println!("{:?}", res2);
        res.push(ImagePredict(file.file_name().unwrap().to_string_lossy().into_owned(),flag));
    }

    println!("mninst model {:?}",x);
    

    Ok(res)
}

fn add() -> Result<(),Box<dyn Error>> {
    let filename = "./addition/model.pb"; // z = x + y
    if !Path::new(filename).exists() {
        return Err(Box::new(
            Status::new_set(
                Code::NotFound,
                &format!(
                    "Run 'python examples/addition/addition.py' to generate {} \
                     and try again.",
                    filename
                ),
            )
            .unwrap(),
        ));
    }
    //入力変数の作成
    let mut x = Tensor::new(&[1]);
    x[0] = 2i32;
    let mut y = Tensor::new(&[1]);
    y[0] = 40i32;

    //python で定義された計算グラフを読み込む
    let mut graph = Graph::new();
    let mut proto = Vec::new();
    File::open(filename)?.read_to_end(&mut proto)?;
    graph.import_graph_def(&proto,&ImportGraphDefOptions::new())?;
    let session = Session::new(&SessionOptions::new(),&graph)?;

    let mut args = SessionRunArgs::new();
    args.add_feed(&graph.operation_by_name_required("x")?,0,&x);
    args.add_feed(&graph.operation_by_name_required("y")?,0,&y);
    let z = args.request_fetch(&graph.operation_by_name_required("z")?,0);
    session.run(&mut args)?;

    let z_res: i32 = args.fetch(z)?[0];
    println!("{:?}",&z_res);

    Ok(())
}

fn main() {
    match add() {
        Ok(()) => println!("add Ok!!"),
        Err(e) => println!("Err :: {}",e),
    }

    match module::cnn() {
        Ok(()) => println!("add Ok!!"),
        Err(e) => println!("Err :: {}",e),
    }

    match color::cnn() {
        Ok(()) => println!("color OK!!!"),
        Err(e) => println!("color Err::{}",e),
    }

    let paths = path::getDirPath("./1").unwrap();
    println!("{:?}",&paths);
    //./cnn/mnist_savedmodel
    //"./examples/mnist_savedmodel"
    let results = model_load("./examples/mnist_savedmodel","./1/testSample").unwrap();
    println!("{}",results[1].0);
    let mut file = File::create("predict.csv").unwrap();

    for result in results {
        println!("{} , {}",result.0,result.1);
        
        file.write_all(format!("{} , {} \n",result.0,result.1).as_bytes());
    }
}
