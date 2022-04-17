def set_tags(args):
    """
    Set wandb tags for the experiment.
    """
    tags = [args.model_name, args.optimizer, args.dset_name]
    if args.custom_layer_name is not None:
        tags.append(args.custom_layer_name)
    if args.custom_tag:
        tags.append(args.custom_tag)
    if args.use_cifar_data_augmentation:
        tags.append("DataAug")
    if args.use_abs and args.custom_layer_name == "NormOut":
        tags.append(f'use-abs')
    if args.custom_layer_name is not None and args.replace_layers is not None:
        tags.append(f"Layers at {args.replace_layers} replaced with {args.custom_layer_name}")
    if args.custom_layer_name is not None and args.insert_layers is not None:
        tags.append(f"{args.custom_layer_name} layers at {args.insert_layers}")
    if args.remove_layers is not None:
        tags.append(f"Layers removed from indices {args.remove_layers}")
    return tags