use rand::distr::Uniform;
use rand::prelude::*;

fn main() {
    let rand_vec = random_vector(0.0, 1.0, 6);
    let empty_linear = Linear::new(3, 2);
}

fn random_vector(low: f32, high: f32, size: usize) -> Vec<f32> {
    let dist = Uniform::new(low, high).unwrap();
    let rng = rand::rng();
    dist.sample_iter(rng).take(size).collect()
}

struct Linear {
    d_in: usize,
    d_out: usize,
    weights: Vec<f32>,
}

impl Linear {
    fn new(d_in: usize, d_out: usize) -> Linear {
        let weights = Vec::with_capacity(d_in * d_out);
        Linear {
            d_in,
            d_out,
            weights,
        }
    }
    fn forward(&self) {}
}
