use rand::distr::Uniform;
use rand::prelude::*;

fn main() {
    let d_in = 3;
    let d_out = 2;
    let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    //let weights = random_vector(0.0, 1.0, d_in * d_out);
    let linear = Linear {
        d_in,
        d_out,
        weights,
    };
    let input = vec![1.0, 1.0, 1.0];
    let output = linear.forward(&input);
    println!("{:?}", output);
    println!("Sigmoid({}) = {}", 0.5, sigmoid(0.5));
    println!("Sigmoid({}) = {}", -30.0, sigmoid(-30.0));
    println!("Sigmoid({}) = {}", 4.5, sigmoid(4.5));
    for y in vec![0.0, 1.0] {
        for x in vec![0.2, 0.4, 0.6, 0.8] {
            println!("BCE(x={},y={}) = {}", x, y, binary_cross_entropy(x, y));
        }
    }
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
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        // Ensure same dimension
        assert!(
            x.len() == self.d_in,
            "Input vector has dim {} incompatible with linear layer input dim {}.",
            x.len(),
            self.d_in
        );

        let mut output: Vec<f32> = Vec::with_capacity(self.d_out);
        for row in 0..self.d_out {
            for col in 0..self.d_in {
                output.push(x[col] * self.weights[row * self.d_in + col]);
            }
        }
        output
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn binary_cross_entropy(x: f32, y: f32) -> f32 {
    assert!(
        x > 0.0 && x < 1.0,
        "Input {} is not a valid probability.",
        x,
    );
    assert!(
        y == 0.0 || y == 1.0,
        "Input {} is not a valid probability.",
        y,
    );
    -(1.0 - y) * (1.0 - x).ln() - y * x.ln()
}
