for perturb in gaussian; do
    for seed in 1234; do #1234 5678 7890 2024 2025 6092 7220 7478 3508 7879
        # echo idood
        python lpn/evaluate_prior_idood.py \
        --model_config_path exps/mnist/configs/mnist/model.json \
        --out_dir exps/mayoct/models_mnist/prior_idood_noAug_1_${seed} \
        --perturb_config_path exps/mnist/configs/mnist/prior/perturb/${perturb}.json \
        --dataset_config_path exps/mnist/configs/mnist/prior/dataset_04.json \
        --model_path exps/mayoct/models_mnist/lpn/s=0.1_noAug_1_7890/model.pt \
        --inv_alg cvx_cg \
        --seed ${seed}
    done
done

