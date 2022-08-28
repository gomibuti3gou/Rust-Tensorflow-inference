use std::error::Error;
use std::fs;
use std::io;
use std::path::Path;
use std::path;

pub fn read_dir<P:AsRef<Path>>(path:P) -> io::Result<Vec<String>> {
  Ok(fs::read_dir(path)?
    .filter_map(|entry| {
      let entry = entry.ok()?;
      if entry.file_type().ok()?.is_file() {
        Some(entry.file_name().to_string_lossy().into_owned())
      } else {
        None
      }
    })
    .collect())
}

pub fn getDirPath(path: &str) -> Result<Vec<path::PathBuf>,Box<dyn Error>> {
  let dir = fs::read_dir(path)?;
  let mut files : Vec<path::PathBuf> = Vec::new();
  for item in dir.into_iter() {
    files.push(item?.path());
  }
  Ok(files)
}