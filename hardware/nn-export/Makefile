all: gen nn-export-test swig

gen:
	python3 gen.py ../../data/models/models_19and20/epoch200_lip3

nn-export-test:
	gcc -std=c99 nn.c main.c -lm -o nn-export-test

swig:
	swig -python nnexport.i
	gcc -std=c99 -shared -fPIC \
		-I/usr/include/python3.8 \
		nnexport_wrap.c \
		nn.c \
		-lm -lpython3.8 \
		-g \
		-o _nnexport.so

clean:
	rm -f generated_weights.c nn-export-test nnexport.py nnexport.pyc _nnexport.so __init__.pyc
