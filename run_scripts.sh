# run training code
!python run_pipline.py --train-dir train_test/train --mode train --batch-size 8 --workers 2 --lr 0.001 --model-path output/model.pth
# run evaluation code
!python run_pipline.py --query-dir train_test/val/query --gallery-dir train_test/val/gallery --mode evaluate --model-path output/model.pth
# run infrence code
python run_pipeline.py --inference --query-image path_to_query.jpg --model-path group_reid_model.pth

