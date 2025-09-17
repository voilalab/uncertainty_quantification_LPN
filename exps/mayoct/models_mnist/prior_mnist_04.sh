
for perturb in convex blur gaussian; do
    python lpn/evaluate_prior.py \
    --model_config_path exps/mnist/configs/mnist/model.json \
    --out_dir exps/mayoct/models_mnist/prior \
    --perturb_config_path exps/mnist/configs/mnist/prior/perturb/${perturb}.json \
    --dataset_config_path exps/mnist/configs/mnist/prior/dataset_04.json \
    --model_path exps/mayoct/models_mnist/lpn/s=0.1/model.pt \
    --inv_alg cvx_cg 
done
