def set_tags(args):
    """
    Set wandb tags for the experiment.
    """
    tags = [args.model_name, args.custom_layer_name, args.optimizer, args.dset_name]
    if args.custom_layer_name != "None" and args.replace_layers is not None:
        tags.append(f"Layers at {args.replace_layers} replaced with {args.custom_layer_name}")
    if args.custom_layer_name != "None" and args.insert_layers is not None:
        tags.append(f"{args.custom_layer_name} layers at {args.insert_layers}")
    if args.remove_layers is not None:
        tags.append(f"Layers removed from indices {args.remove_layers}")
    if args.custom_tag:
        tags.append(args.custom_tag)
    if args.use_cifar_data_augmentation:
        tags.append("DataAug")
    if args.custom_layer_name == "NormOut":
        tags.append(args.normout_method)
    if args.normout_method == "exp":
        tags.append(f'exponent={args.exponent}')
    return tags