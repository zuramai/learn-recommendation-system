fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=../proto/recommender.proto");
    tonic_prost_build::configure()
        .build_server(false)
        .compile_protos(&["../proto/recommender.proto"], &["../proto"])?;
    Ok(())
}
