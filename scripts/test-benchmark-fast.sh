echo 'set5' &&
echo 'x2' &&
python test.py --config ./configs/hiiftest/test-fast-set5-2.yaml --fast True --model $1 --gpu $2 &&
echo 'x3' &&
python test.py --config ./configs/hiiftest/test-fast-set5-3.yaml --fast True --model $1 --gpu $2 &&
echo 'x4' &&
python test.py --config ./configs/hiiftest/test-fast-set5-4.yaml --fast True --model $1 --gpu $2 &&
echo 'x6*' &&
python test.py --config ./configs/hiiftest/test-fast-set5-6.yaml --fast True --model $1 --gpu $2 &&
echo 'x8*' &&
python test.py --config ./configs/hiiftest/test-fast-set5-8.yaml --fast True --model $1 --gpu $2 &&
echo 'x12*' &&
python test.py --config ./configs/hiiftest/test-fast-set5-12.yaml --fast True --model $1 --gpu $2 &&

echo 'set14' &&
echo 'x2' &&
python test.py --config ./configs/hiiftest/test-fast-set14-2.yaml --fast True --model $1 --gpu $2 &&
echo 'x3' &&
python test.py --config ./configs/hiiftest/test-fast-set14-3.yaml --fast True --model $1 --gpu $2 &&
echo 'x4' &&
python test.py --config ./configs/hiiftest/test-fast-set14-4.yaml --fast True --model $1 --gpu $2 &&
echo 'x6*' &&
python test.py --config ./configs/hiiftest/test-fast-set14-6.yaml --fast True --model $1 --gpu $2 &&
echo 'x8*' &&
python test.py --config ./configs/hiiftest/test-fast-set14-8.yaml --fast True --model $1 --gpu $2 &&
echo 'x12*' &&
python test.py --config ./configs/hiiftest/test-fast-set14-12.yaml --fast True --model $1 --gpu $2 &&

echo 'b100' &&
echo 'x2' &&
python test.py --config ./configs/hiiftest/test-fast-b100-2.yaml --fast True --model $1 --gpu $2 &&
echo 'x3' &&
python test.py --config ./configs/hiiftest/test-fast-b100-3.yaml --fast True --model $1 --gpu $2 &&
echo 'x4' &&
python test.py --config ./configs/hiiftest/test-fast-b100-4.yaml --fast True --model $1 --gpu $2 &&
echo 'x6*' &&
python test.py --config ./configs/hiiftest/test-fast-b100-6.yaml --fast True --model $1 --gpu $2 &&
echo 'x8*' &&
python test.py --config ./configs/hiiftest/test-fast-b100-8.yaml --fast True --model $1 --gpu $2 &&
echo 'x12*' &&
python test.py --config ./configs/hiiftest/test-fast-b100-12.yaml --fast True --model $1 --gpu $2 &&

echo 'urban100' &&
echo 'x2' &&
python test.py --config ./configs/hiiftest/test-fast-urban100-2.yaml --fast True --model $1 --gpu $2 &&
echo 'x3' &&
python test.py --config ./configs/hiiftest/test-fast-urban100-3.yaml --fast True --model $1 --gpu $2 &&
echo 'x4' &&
python test.py --config ./configs/hiiftest/test-fast-urban100-4.yaml --fast True --model $1 --gpu $2 &&
echo 'x6*' &&
python test.py --config ./configs/hiiftest/test-fast-urban100-6.yaml --fast True --model $1 --gpu $2 &&
echo 'x8*' &&
python test.py --config ./configs/hiiftest/test-fast-urban100-8.yaml --fast True --model $1 --gpu $2 &&
echo 'x12*' &&
python test.py --config ./configs/hiiftest/test-fast-urban100-12.yaml --fast True --model $1 --gpu $2 &&


#echo 'manga109' &&
#echo 'x2' &&
#python test.py --config ./configs/hiiftest/test-fast-manga109-2.yaml --fast True --model $1 --gpu $2 &&
#echo 'x3' &&
#python test.py --config ./configs/hiiftest/test-fast-manga109-3.yaml --fast True --model $1 --gpu $2 &&
#echo 'x4' &&
#python test.py --config ./configs/hiiftest/test-fast-manga109-4.yaml --fast True --model $1 --gpu $2 &&
#echo 'x6*' &&
#python test.py --config ./configs/hiiftest/test-fast-manga109-6.yaml --fast True --model $1 --gpu $2 &&
#echo 'x8*' &&
#python test.py --config ./configs/hiiftest/test-fast-manga109-8.yaml --fast True --model $1 --gpu $2 &&
#echo 'x12*' &&
#python test.py --config ./configs/hiiftest/test-fast-manga109-12.yaml --fast True --model $1 --gpu $2 &&
true
