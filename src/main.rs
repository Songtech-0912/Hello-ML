//! Demonstration of MNIST digits classification with multi-layer-perceptron
//! originally from https://github.com/raskr/rust-autograd with modifications
use autograd as ag;
use autograd::ndarray;

use ag::optimizers::Adam;
use ag::prelude::*;
use ag::rand::seq::SliceRandom;
use ag::tensor_ops as T;

use ag::{ndarray_ext as array, Context};
use indicatif::ProgressBar;
use indicatif::ProgressStyle;
use ndarray::s;
use std::env;
use std::ops::Deref;
use std::time::Instant;

mod mnist_data;

type Tensor<'graph> = ag::Tensor<'graph, f32>;

fn inputs<'g>(g: &'g Context<f32>) -> (Tensor<'g>, Tensor<'g>) {
    let x = g.placeholder("x", &[-1, 28 * 28]);
    let y = g.placeholder("y", &[-1, 1]);
    (x, y)
}

fn get_permutation(size: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..size).collect();
    perm.shuffle(&mut ag::ndarray_ext::get_default_rng());
    perm
}

fn main() {
    let debug_mode: bool = match env::var("ML_DEBUG_TRUE") {
        Ok(_) => true,
        Err(_) => false,
    };
    let ((x_train, y_train), (x_test, y_test)) = mnist_data::load();

    let rng = ag::ndarray_ext::ArrayRng::<f32>::default();

    let mut env = ag::VariableEnvironment::new();
    let w = env.set(rng.glorot_uniform(&[28 * 28, 10]));
    let b = env.set(array::zeros(&[1, 10]));

    let adam = Adam::default("adam", env.default_namespace().current_var_ids(), &mut env);

    let max_epoch = 50;
    let batch_size: isize = 200;
    let num_samples = x_train.shape()[0];
    let num_batches = num_samples / batch_size as usize;

    for epoch in 0..max_epoch {
        let mut loss_sum = 0.0;
        let start = Instant::now();
        let bar = ProgressBar::new(num_batches.try_into().unwrap());
        bar.set_style(ProgressStyle::default_bar().progress_chars("==."));

        for i in get_permutation(num_batches) {
            bar.inc(1);
            let i = i as isize * batch_size;
            let x_batch = x_train.slice(s![i..i + batch_size, ..]).into_dyn();
            let y_batch = y_train.slice(s![i..i + batch_size, ..]).into_dyn();

            env.run(|ctx| {
                let w = ctx.variable(w);
                let b = ctx.variable(b);
                let (x, y) = inputs(ctx.deref());
                let z = T::matmul(x, w) + b;
                let loss = T::sparse_softmax_cross_entropy(z, &y);
                let mean_loss = T::reduce_mean(loss, &[0], false);
                let grads = &T::grad(&[&mean_loss], &[w, b]);
                let mut feeder = ag::Feeder::new();
                feeder.push(x, x_batch).push(y, y_batch);
                adam.update(&[w, b], grads, ctx, feeder);
                if debug_mode {
                    let eval_loss = ctx
                        .evaluator()
                        .push(mean_loss)
                        .feed(x, x_test.view())
                        .feed(y, y_test.view())
                        .run();
                    loss_sum += eval_loss[0].as_ref().unwrap()[0];
                }
            });
        }
        bar.finish();
        let end = start.elapsed();
        if debug_mode {
            println!(
                "Finish epoch {} in {}.{:03} sec with loss {}",
                epoch,
                end.as_secs(),
                end.subsec_nanos() / 1_000_000,
                loss_sum / num_batches as f32,
            );
        } else {
            println!(
                "Finish epoch {} in {}.{:03} sec with loss",
                epoch,
                end.as_secs(),
                end.subsec_nanos() / 1_000_000
            );
        }
    }

    // Predict
    env.run(|c| {
        let w = c.variable(w);
        let b = c.variable(b);
        let (x, y) = inputs(c.deref());

        // -- test --
        let z = T::matmul(x, w) + b;
        let loss = T::sparse_softmax_cross_entropy(z, &y);
        let predictions = T::argmax(z, -1, true);
        let accuracy = T::reduce_mean(&T::equal(predictions, &y), &[0, 1], false);
        let mean_loss = T::reduce_mean(&T::equal(loss, &y), &[0, 1], false);
        println!(
            "Final test accuracy: {:?}",
            c.evaluator()
                .push(accuracy)
                .feed(x, x_test.view())
                .feed(y, y_test.view())
                .run()
        );
        println!(
            "Final mean loss: {:?}",
            c.evaluator()
                .push(mean_loss)
                .feed(x, x_test.view())
                .feed(y, y_test.view())
                .run()
        );
    })
}
