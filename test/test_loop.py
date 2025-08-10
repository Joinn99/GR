def all_merging(args, model_merger_class):
    eval_groups = {}
    all_domains = ["Video_Games", "Movies_and_TV", "Cell_Phones_and_Accessories", "Sports_and_Outdoors", "Books"]
    for source_domain in all_domains:
        args.source_domain = source_domain
        eval_groups[source_domain] = []
        target_domains = [domain for domain in all_domains if domain != source_domain]
        args.target_domains = target_domains
        for method in ["average_merging", "ties_merging", "mask_merging", "task_arithmetic"]:
            args.method = method
            try:
                eval_groups[source_domain].append(("merged", model_merger_class(args).run()))
            except Exception as e:
                print(f"Error during merging: {e}")
                continue
    return eval_groups

def add_one_merging(args, model_merger_class):
    eval_groups = {}
    for source_domain in ["Video_Games", "Movies_and_TV", "Cell_Phones_and_Accessories", "Sports_and_Outdoors", "Books"]:
        args.source_domain = source_domain
        eval_groups[source_domain] = []
        for target_domains in [["Video_Games"], ["Movies_and_TV"], ["Cell_Phones_and_Accessories"], ["Sports_and_Outdoors"], ["Books"]]:
            args.target_domains = target_domains
            if source_domain in target_domains:
                continue
            for method in ["average_merging", "ties_merging", "mask_merging", "task_arithmetic"]:
                args.method = method
                try:
                    eval_groups[source_domain].append(("merged", model_merger_class(args).run()))
                except Exception as e:
                    print(f"Error during merging: {e}")
                    continue
    return eval_groups


def get_eval_groups(name = "all_merging"):
    if name == "all_merging":
        return all_merging
    elif name == "add_one_merging":
        return add_one_merging
    else:
        raise ValueError(f"Invalid name: {name}")