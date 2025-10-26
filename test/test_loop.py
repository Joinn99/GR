import torch

TEST_DOMAINS = ["Video_Games", "Movies_and_TV", "Cell_Phones_and_Accessories", "Sports_and_Outdoors", "Books", "Office_Products", "Health_and_Household", "Software"]

def single_run(args, model_merger_class):
    merger = model_merger_class(args)
    output_name = merger.run()
    del merger
    torch.cuda.empty_cache()
    return output_name

def all_merging(args, model_merger_class):
    eval_groups = {}
    all_domains = TEST_DOMAINS
    for source_domain in all_domains:
        args.source_domain = source_domain
        eval_groups[source_domain] = []
        target_domains = [domain for domain in all_domains if domain != source_domain]
        args.target_domains = target_domains
        for method in ["average_merging", "ties_merging", "mask_merging", "task_arithmetic"]:
            args.method = method
            try:
                eval_groups[source_domain].append(("merged", single_run(args, model_merger_class)))
            except Exception as e:
                print(f"Error during merging: {e}")
                continue
    return eval_groups

def add_one_merging(args, model_merger_class):
    eval_groups = {}
    for source_domain in TEST_DOMAINS:
        args.source_domain = source_domain
        eval_groups[source_domain] = []
        for target_domains in [[domain] for domain in TEST_DOMAINS if domain != source_domain]:
            args.target_domains = target_domains
            for method in ["average_merging", "ties_merging", "mask_merging", "task_arithmetic"]:
                args.method = method
                try:
                    eval_groups[source_domain].append(("merged", single_run(args, model_merger_class)))
                except Exception as e:
                    print(f"Error during merging: {e}")
                    continue
    return eval_groups

def temporal_task_arithmetic_merging(args, model_merger_class):
    eval_groups = {args.source_domain: []}
    import numpy as np
    scaling_coefficient_list = np.arange(-0.5, 0.5, 0.005)
    for scaling_coefficient in scaling_coefficient_list:
        args.merging_args = {"scaling_coefficient": scaling_coefficient}
        eval_groups[args.source_domain].append(("merged", single_run(args, model_merger_class)))
    return eval_groups

def single_test_merging(args, model_merger_class):
    eval_groups = {args.source_domain: [("merged", single_run(args, model_merger_class))]}
    return eval_groups

def complete_merging(args, model_merger_class):
    eval_groups = {}
    for source_domain in TEST_DOMAINS:
        args.source_domain = source_domain
        eval_groups[source_domain] = []
        for target_domains in [[domain] for domain in TEST_DOMAINS if domain != source_domain]:
            args.target_domains = target_domains
            args.method = "mask_merging" if args.mode == "title" else "ties_merging"
            try:
                eval_groups[source_domain].append(("merged", single_run(args, model_merger_class)))
            except Exception as e:
                print(f"Error during merging: {e}")
    return eval_groups

def get_eval_groups(name = "all_merging"):
    if name == "all_merging":
        return all_merging
    elif name == "add_one_merging":
        return add_one_merging
    elif name == "single_test_merging":
        return single_test_merging
    elif name == "complete_merging":
        return complete_merging
    elif name == "temporal_task_arithmetic_merging":
        return temporal_task_arithmetic_merging
    else:
        raise ValueError(f"Invalid name: {name}")