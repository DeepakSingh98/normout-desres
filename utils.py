def set_tags(args):
    """
    Set wandb tags for the experiment.
    """
    tags = [args.model_name, args.optimizer, args.dset_name]
    if args.custom_layer_name is not None:
        tags.append(args.custom_layer_name)
        if args.custom_layer_name == "Dropout":
            tags.append(f"p_{str(args.dropout_p)}")
        elif args.custom_layer_name == "TopK":
            tags.append(f"k_{str(args.topk_k)}")
    if args.custom_tag:
        tags.append(args.custom_tag)
    if args.pretrained:
        tags.append("pretrained")
    if not args.no_data_augmentation:
        tags.append("DataAug")
    if not args.no_abs and args.custom_layer_name == "NormOut":
        tags.append(f'use-abs')
    if args.custom_layer_name == "NormOut":
        tags.append(f"{args.max_type}")
    if (args.custom_layer_name == "NormOut" or args.custom_layer_name == "Dropout") and args.on_at_inference:
        tags.append("on-at-inference")
    if args.custom_layer_name is not None and args.replace_layers is not None:
        tags.append(f"Layers at {args.replace_layers} replaced with {args.custom_layer_name}")
    if args.custom_layer_name is not None and args.insert_layers is not None:
        tags.append(f"{args.custom_layer_name} layers at {','.join([str(i) for i in args.insert_layers])}")
    if args.remove_layers is not None:
        tags.append(f"Layers removed from indices {args.remove_layers}")
    if not args.no_pgd:
        tags.append(f"pgd steps: {args.pgd_steps}")
    return tags