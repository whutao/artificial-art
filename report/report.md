---
title: "AI assignment 2 report"
author: "Roman Nabiullin"
date: "12/04/2021"
output: html_document
---

<style type="text/css">body{ font-size: 14pt; }</style>

# Report

## 1. Image processing, CUDA and optimization

An image of *.jpg* or *.png* format is opened and converted into a NumPy array. The size of this array is 

$$\text{IMAGE\_HEIGHT} \times \text{IMAGE\_WIDTH} \times 3$$

where $3$ stands for RGB channels.

It was decided to use python library `numba` that provides with just-in-time compilation and CUDA facility. This allows to increase the performance dramatically - reduce from several hours per standard epoch (*epoch* is decribed in the next section) to $3-6$ minutes per standard epoch.

Reference links:
* [CUDA](https://docs.nvidia.com/cuda/)
* [numba](https://numba.pydata.org/)
* [Just-in-time compilation](https://en.wikipedia.org/wiki/Just-in-time_compilation)

## 2.1 Algorithm description

```
function run_evolution(inital_popualtion, epoch_duration, epoch_count) {
    population <- initial_population;

    for epoch_id from 1 to epoch_count {
        for generation_id from 1 to epoch_duration {
            population <- run_generation(population);
        }

        save epoch snapshot;
    }
}

function run_generation(current_generation) {
    next_generation <- empty list;
    best <- individual from current_generation with the best fitness;
    add best to next_generation;
    repeat {
        successor <- selection(current_generation);
        successor <- mutation(successor);
        add successor to next_generation;
    } until next_generation contatins exactly population_size individuals;

    return next_generation;
}
```

It was decided to use the above evolutionary algorithm.

Since in most cases, it is almost impossible to observe any difference after a single mutation, we introduce a concept of *epoch*. Epoch - is a batch of, for instance, 10k generations (as in *standard epoch*). Snapshots including best-fit chromosomes and best-fitnesses are taken and saved after each epoch.

## 2.2 Chromosome representation

In our case, a gene is a triangle of some shape and color. A chromosome is represented as an ordered sequence of genes. 

It was decided to maintain a chromosome as a composed NumPy array for the sake of efficiency. 

## 2.3 Fitness function

Given two images of the same size $\text{H}\times \text{W} \times 3$. The distance between these two images is going to be the the double sum

$$\sum_{i=1}^{H} \sum_{j=1}^{W} \frac{\sqrt{\Delta R_{ij}^2 + \Delta G_{ij}^2 + \Delta B_{ij}^2}}{\sqrt{255^2 + 255^2 + 255^2}}$$

where $\Delta R_{ij}, \Delta G_{ij}, \Delta B_{ij}$ are difference for each of RGB channel for $i, j$ pixel. Basically, it is the sum of ratios of Euclidean distance between $2$ pixels and the maximal Euclidean distance between $2$ pixels. The range for fitness values $\left[ 0 \dots 3HW \right]$.

So, if the fitness value is small then two images are rather similar. If fitness value is big, two images are completely different.

As for the fitness limit, we can set it to 

$$0.08 \cdot \text{max\_fitness} = 0.24 H W$$

Tests demonstrate that $8\%$ of total difference is enough for a human (at least me) to call two images pretty similar.

## 2.4 Selection

Roulette wheel selection method has been chosen. As for the weights, we use the following approach

$$\text{if fitnesses are } f = \left[f_1, \dots f_n \right] \text{ then respective probabilities are }$$

$$p = \left[\frac{f_1^{-1}}{\sum f^{-1}}, \dots \frac{f_n^{-1}}{\sum f^{-1}} \right] \text{ where } f^{-1} = \left[ f_1^{-1}, \dots f_n^{-1} \right]$$

We should use inverted fitness values to calculate probabilities because of the nature of our fitness function (section 2.3).

## 2.5 Crossover

It was decided not to use any crossover function.

## 2.6 Mutation

## 3. Examples

## 4. Answer: What is art for me?