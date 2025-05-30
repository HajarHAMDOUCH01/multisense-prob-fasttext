/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *               2018-present, Ben Athiwaratkun
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef FASTTEXT_ARGS_H
#define FASTTEXT_ARGS_H

#include <istream>
#include <ostream>
#include <string>
#include "matrix.h"
namespace fasttext {

enum class model_name : int {cbow=1, sg, sup};
enum class loss_name : int {hs=1, ns, softmax};

class Args {
  public:
    Args();
    std::string input;
    std::string test;
    std::string output;
    real diversity_weight;
    double lr;
    int lrUpdateRate;
    int dim;
    int ws;
    int epoch;
    int minCount;
    int minCountLabel;
    int neg;
    int wordNgrams;
    loss_name loss;
    model_name model;
    int bucket;
    int minn;
    int maxn;
    int thread;
    double t;
    std::string label;
    int verbose;
    std::string pretrainedVectors;
    int saveOutput;

    bool qout;
    bool retrain;
    bool qnorm;
    size_t cutoff;
    size_t dsub;

    void parseArgs(int, char**);
    void printHelp();
    void save(std::ostream&);
    void load(std::istream&);

    // BenA
    double gs_lambda;
    int num_gs_samples;
    int include_dictemb;
    int add_dictemb;
    int drop_sub;
    int drop_dict;

    float gs_subword;
    int num_subgs_samples;
    // BenA: for max margin
    float margin;
    float var_scale;
    bool multi;
    bool expdot;
    bool var;
};

}

#endif
