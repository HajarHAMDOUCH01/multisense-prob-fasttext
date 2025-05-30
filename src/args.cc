/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *               2018-present, Ben Athiwaratkun
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "args.h"

#include <stdlib.h>
#include <string.h>

#include <iostream>

namespace fasttext {

Args::Args() {
  diversity_weight = 0.0;
  lr = 0.001;
  dim = 100;
  ws = 100;
  epoch = 5;
  minCount = 5;
  minCountLabel = 0;
  neg = 1;
  wordNgrams = 1;
  loss = loss_name::ns;
  model = model_name::sg;
  bucket = 2000000;
  minn = 3;
  maxn = 6;
  thread = 12;
  lrUpdateRate = 100;
  t = 1e-4;
  label = "__label__";
  verbose = 2;
  pretrainedVectors = "";
  saveOutput = 0;

  qout = false;
  retrain = false;
  qnorm = false;
  cutoff = 0;
  dsub = 2;
  gs_lambda = 0.0;
  num_gs_samples = 0;
  include_dictemb = 1; // the default behavior is to include w_dict in the average
  add_dictemb = 0; // by default, dictemb is not added outside
  // For dropout
  drop_sub = 0;
  drop_dict = 0;
  // Group Sparsity within the subword
  gs_subword = 0.0;
  num_subgs_samples = 10;

  // BenA: for max margin loss
  margin = 1.0;
  var_scale = 0.05; // This is the default for 50 dim - find a good value for 300 dim
  multi = true;
  expdot = false;
  var = false;
}

void Args::parseArgs(int argc, char** argv) {

  std::string command(argv[1]);
  if (command == "supervised") {
    model = model_name::sup;
    loss = loss_name::softmax;
    minCount = 1;
    minn = 0;
    maxn = 0;
    lr = 0.1;
  } else if (command == "cbow") {
    model = model_name::cbow;
  }

  int ai = 2;
  while (ai < argc) {
    if (argv[ai][0] != '-') {
      std::cerr << "Provided argument without a dash! Usage:" << std::endl;
      printHelp();
      exit(EXIT_FAILURE);
    }
    // All known arguments should be chained using 'else if'
    else if (strcmp(argv[ai], "-h") == 0) {
      std::cerr << "Here is the help! Usage:" << std::endl;
      printHelp();
      exit(EXIT_FAILURE);
    } else if (strcmp(argv[ai], "-input") == 0) {
      input = std::string(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-test") == 0) {
      test = std::string(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-output") == 0) {
      output = std::string(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-lr") == 0) {
      lr = atof(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-lrUpdateRate") == 0) {
      lrUpdateRate = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-dim") == 0) {
      dim = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-ws") == 0) {
      ws = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-epoch") == 0) {
      epoch = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-minCount") == 0) {
      minCount = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-minCountLabel") == 0) {
      minCountLabel = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-neg") == 0) {
      neg = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-wordNgrams") == 0) {
      wordNgrams = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-loss") == 0) {
      if (strcmp(argv[ai + 1], "hs") == 0) {
        loss = loss_name::hs;
      } else if (strcmp(argv[ai + 1], "ns") == 0) {
        loss = loss_name::ns;
      } else if (strcmp(argv[ai + 1], "softmax") == 0) {
        loss = loss_name::softmax;
      } else {
        std::cerr << "Unknown loss: " << argv[ai + 1] << std::endl;
        printHelp();
        exit(EXIT_FAILURE);
      }
    } else if (strcmp(argv[ai], "-bucket") == 0) {
      bucket = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-minn") == 0) {
      minn = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-maxn") == 0) {
      maxn = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-thread") == 0) {
      thread = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-t") == 0) {
      t = atof(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-label") == 0) {
      label = std::string(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-verbose") == 0) {
      verbose = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-pretrainedVectors") == 0) {
      pretrainedVectors = std::string(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-saveOutput") == 0) {
      saveOutput = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-qnorm") == 0) {
      qnorm = true; ai--; // These decrement 'ai' because they don't have a value argument
    } else if (strcmp(argv[ai], "-retrain") == 0) {
      retrain = true; ai--;
    } else if (strcmp(argv[ai], "-qout") == 0) {
      qout = true; ai--;
    } else if (strcmp(argv[ai], "-cutoff") == 0) {
      cutoff = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-dsub") == 0) {
      dsub = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-gs_lambda") == 0) {
      gs_lambda = atof(argv[ai + 1]);
      std::cerr << "Group Sparsity Lambda = " << gs_lambda << std::endl;
    } else if (strcmp(argv[ai], "-num_gs_samples") == 0) {
      num_gs_samples = atoi(argv[ai + 1]);
      std::cerr << "Number of Group Sparsity Samples = " << num_gs_samples << std::endl;
    } else if (strcmp(argv[ai], "-include_dictemb") == 0) {
      include_dictemb = atoi(argv[ai + 1]);
      std::cerr << "Setting: including dict embedding in subword vector averaging =" << include_dictemb << std::endl;
    } else if (strcmp(argv[ai], "-add_dictemb") == 0) {
      add_dictemb = atoi(argv[ai + 1]);
      std::cerr << "Setting: add dict embedding outside of subword vector averaging =" << add_dictemb << std::endl;
    } else if (strcmp(argv[ai], "-drop_sub") == 0) {
      drop_sub = atoi(argv[ai + 1]);
      std::cerr << "Dropping subword embeddings = " << drop_sub << std::endl;
    } else if (strcmp(argv[ai], "-drop_dict") == 0) {
      drop_dict = atoi(argv[ai + 1]);
      std::cerr << "Dropping dict embeddings = " << drop_dict << std::endl;
    } else if (strcmp(argv[ai], "-gs_subword") == 0) {
      gs_subword = atof(argv[ai + 1]);
      std::cerr << "Group Sparsity Strength within subword " << gs_subword << std::endl;
    } else if (strcmp(argv[ai], "-num_subgs_samples") == 0) {
      num_subgs_samples = atoi(argv[ai + 1]);
      std::cerr << "Num subgs samples" << num_subgs_samples << std::endl;
    } else if (strcmp(argv[ai], "-margin") == 0) {
      margin = atof(argv[ai + 1]);
      std::cerr << "Margin" << margin << std::endl;
    } else if (strcmp(argv[ai], "-multi") == 0) {
      multi = atoi(argv[ai + 1]); // 0 for false and else for true
      std::cerr << "Multi" << multi << std::endl;
    } else if (strcmp(argv[ai], "-var_scale") == 0) {
      var_scale = atof(argv[ai + 1]); // 0 for false and else for true
      std::cerr << "var scale" << var_scale << std::endl;
    } else if (strcmp(argv[ai], "-expdot") == 0) {
      expdot = atoi(argv[ai + 1]); // 0 for false and else for true
      std::cerr << "expdot" << expdot << std::endl;
    } else if (strcmp(argv[ai], "-var") == 0) {
      var = atoi(argv[ai + 1]); // 0 for false and else for true
      std::cerr << "var" << var << std::endl;
    }
    // THIS IS THE CORRECT PLACE FOR THE NEW ARGUMENT
    else if (strcmp(argv[ai], "-diversity_weight") == 0) {
      diversity_weight = atof(argv[ai + 1]);
      std::cerr << "Diversity Weight: " << diversity_weight << std::endl;
    }
    // The final 'else' should catch all unrecognized arguments
    else {
      std::cerr << "Unknown argument: " << argv[ai] << std::endl;
      printHelp();
      exit(EXIT_FAILURE);
    }
    ai += 2; // Increment by 2 for argument flag and its value
  }

  if (input.empty() || output.empty()) {
    std::cerr << "Empty input or output path." << std::endl;
    printHelp();
    exit(EXIT_FAILURE);
  }
  if (wordNgrams <= 1 && maxn == 0) {
    bucket = 0;
  }
}

void Args::printHelp() {
  std::string lname = "ns";
  if (loss == loss_name::hs) lname = "hs";
  if (loss == loss_name::softmax) lname = "softmax";
  std::cerr
    << "\nThe following arguments are mandatory:\n"
    << "  -input              training file path\n"
    << "  -output             output file path\n"
    << "\nThe following arguments are optional:\n"
    << "  -verbose            verbosity level [" << verbose << "]\n"
    << "\nThe following arguments for the dictionary are optional:\n"
    << "  -minCount           minimal number of word occurences [" << minCount << "]\n"
    << "  -minCountLabel      minimal number of label occurences [" << minCountLabel << "]\n"
    << "  -wordNgrams         max length of word ngram [" << wordNgrams << "]\n"
    << "  -bucket             number of buckets [" << bucket << "]\n"
    << "  -minn               min length of char ngram [" << minn << "]\n"
    << "  -maxn               max length of char ngram [" << maxn << "]\n"
    << "  -t                  sampling threshold [" << t << "]\n"
    << "  -label              labels prefix [" << label << "]\n"
    << "\nThe following arguments for training are optional:\n"
    << "  -lr                 learning rate [" << lr << "]\n"
    << "  -lrUpdateRate       change the rate of updates for the learning rate [" << lrUpdateRate << "]\n"
    << "  -dim                size of word vectors [" << dim << "]\n"
    << "  -ws                 size of the context window [" << ws << "]\n"
    << "  -epoch              number of epochs [" << epoch << "]\n"
    << "  -neg                number of negatives sampled [" << neg << "]\n"
    << "  -loss               loss function {ns, hs, softmax} [ns]\n"
    << "  -thread             number of threads [" << thread << "]\n"
    << "  -pretrainedVectors  pretrained word vectors for supervised learning ["<< pretrainedVectors <<"]\n"
    << "  -saveOutput         whether output params should be saved [" << saveOutput << "]\n"
    << "\nThe following arguments for quantization are optional:\n"
    << "  -cutoff             number of words and ngrams to retain [" << cutoff << "]\n"
    << "  -retrain            finetune embeddings if a cutoff is applied [" << retrain << "]\n"
    << "  -qnorm              quantizing the norm separately [" << qnorm << "]\n"
    << "  -qout               quantizing the classifier [" << qout << "]\n"
    << "  -dsub               size of each sub-vector [" << dsub << "]\n"
    << std::endl;
}

void Args::save(std::ostream& out) {
  out.write((char*) &(dim), sizeof(int));
  out.write((char*) &(ws), sizeof(int));
  out.write((char*) &(epoch), sizeof(int));
  out.write((char*) &(minCount), sizeof(int));
  out.write((char*) &(neg), sizeof(int));
  out.write((char*) &(wordNgrams), sizeof(int));
  out.write((char*) &(loss), sizeof(loss_name));
  out.write((char*) &(model), sizeof(model_name));
  out.write((char*) &(bucket), sizeof(int));
  out.write((char*) &(minn), sizeof(int));
  out.write((char*) &(maxn), sizeof(int));
  out.write((char*) &(lrUpdateRate), sizeof(int));
  out.write((char*) &(t), sizeof(double));
}

void Args::load(std::istream& in) {
  in.read((char*) &(dim), sizeof(int));
  in.read((char*) &(ws), sizeof(int));
  in.read((char*) &(epoch), sizeof(int));
  in.read((char*) &(minCount), sizeof(int));
  in.read((char*) &(neg), sizeof(int));
  in.read((char*) &(wordNgrams), sizeof(int));
  in.read((char*) &(loss), sizeof(loss_name));
  in.read((char*) &(model), sizeof(model_name));
  in.read((char*) &(bucket), sizeof(int));
  in.read((char*) &(minn), sizeof(int));
  in.read((char*) &(maxn), sizeof(int));
  in.read((char*) &(lrUpdateRate), sizeof(int));
  in.read((char*) &(t), sizeof(double));
}

}
