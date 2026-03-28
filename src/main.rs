use rand::distr::Uniform;
use rand::prelude::*;

fn main() {
    let d_in = 10;
    let d_out = 1;
    // let weights = vec![1.0, 2.0, -0.5, -1.2, 0.0];
    // TODO: use some non-random weight initialization
    let weights = random_vector(-5.0, 5.0, d_in * d_out);
    let mut linear = Linear {
        d_in,
        d_out,
        weights,
    };

    let input = random_vector(-5.0, 5.0, d_in);
    // let input = vec![1.0, 1.0, 1.0, 1.0, 1.0];
    let label = 0.0;
    println!("Weights at begin: {:?}", linear.weights);
    optimize_linear(&mut linear, &input, label, 70);
    println!("Weights at end: {:?}", linear.weights);

    // let output = linear.forward(&input);
    // let output_scalar = output[0];
    // println!("Output of linear layer: {:?}", output_scalar);
    // let logit = sigmoid(output_scalar);
    // println!("Sigmoid of output: {}", logit);
    // let loss = binary_cross_entropy(logit, label);
    // println!("BCE(x={},y={}) = {}", logit, label, loss);
    // let bce_grad = bce_grad(logit, label);
    // println!("BCE grad = {}", bce_grad);
    //
    // let derivative = sigmoid_grad(logit);
    // println!("Derivative: {}", derivative);
}

fn random_vector(low: f32, high: f32, size: usize) -> Vec<f32> {
    let dist = Uniform::new(low, high).unwrap();
    let rng = rand::rng();
    dist.sample_iter(rng).take(size).collect()
}

fn optimize_linear(linear: &mut Linear, input: &[f32], label: f32, iterations: u32) {
    // Optimizes for one datapoint
    for i in 0..iterations {
        let output = linear.forward(&input);
        let logits: Vec<f32> = output.iter().map(|x| sigmoid(x.clone())).collect();
        println!("{:?}", logits);
        let losses: Vec<f32> = logits
            .iter()
            .map(|x| binary_cross_entropy(x.clone(), label))
            .collect();

        let loss_grads: Vec<f32> = logits.iter().map(|x| bce_grad(x.clone(), label)).collect();
        let mut sig_grads: Vec<f32> = Vec::with_capacity(output.len());
        for j in 0..loss_grads.len() {
            sig_grads.push(loss_grads[j] * sigmoid_grad(output[j]));
        }
        let linear_grads = linear.backward(input, &sig_grads, 0.01);
    }
}

// Would be really cool if we had a general Layer, that has input/output dimensions, needs to have
// forward and backward pass
struct Linear {
    d_in: usize,
    d_out: usize,
    weights: Vec<f32>,
}

impl Linear {
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        // Forward pass on single sample
        // TODO: make this work on N samples
        assert!(
            x.len() == self.d_in,
            "Input vector has dim {} incompatible with linear layer input dim {}.",
            x.len(),
            self.d_in
        );

        let mut output: Vec<f32> = vec![0.0; self.d_out];
        for row in 0..self.d_out {
            for col in 0..self.d_in {
                output[row] += x[col] * self.weights[row * self.d_in + col];
            }
        }
        output
    }

    fn backward(&mut self, x: &[f32], grad: &[f32], lr: f32) -> Vec<f32> {
        // Backward pass on single sample
        // TODO: make this work on N samples
        assert!(
            x.len() == self.d_in,
            "Input vector has dim {} incompatible with linear layer input dim {}.",
            x.len(),
            self.d_in
        );
        assert!(
            grad.len() == self.d_out,
            "Grad vector has dim {} incompatible with linear layer output dim {}.",
            grad.len(),
            self.d_out
        );

        let mut back_grad = vec![0.0; self.d_in];
        for row in 0..self.d_out {
            for col in 0..self.d_in {
                back_grad[col] += self.weights[row * self.d_in + col] * grad[row];
                self.weights[row * self.d_in + col] -= lr * x[col] * grad[row];
            }
        }
        back_grad
    }
}

struct Sigmoid {}

impl Sigmoid {
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        x.to_vec().into_iter().map(|x| sigmoid(x)).collect()
    }
    fn backward(&self, grad: &[f32]) -> Vec<f32> {
        grad.to_vec().into_iter().map(|x| sigmoid_grad(x)).collect()
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_grad(x: f32) -> f32 {
    sigmoid(x) * (1.0 - sigmoid(x))
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

fn bce_grad(x: f32, y: f32) -> f32 {
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
    (1.0 - y) / (1.0 - x) - y / x
}
