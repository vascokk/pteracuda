#include <stdio.h>
#include "pcuda_buffer.h"
#include "pcuda_ops.h"

PCudaMatrixFloatBuffer::PCudaMatrixFloatBuffer() {
    this->data = new std::vector<double>();
    this->_rows = 1;
}

PCudaMatrixFloatBuffer::PCudaMatrixFloatBuffer(unsigned  rows, unsigned  cols) {
    this->_rows = rows;
    this->_cols = cols;
    this->data = new std::vector<double>(rows*cols);
}

PCudaMatrixFloatBuffer::~PCudaMatrixFloatBuffer() {
    delete this->data;
}

unsigned int PCudaMatrixFloatBuffer::size() {
    return this->data->size();
}

void PCudaMatrixFloatBuffer::write(ErlNifEnv *env, ERL_NIF_TERM data) {
    ERL_NIF_TERM head_row;
    ERL_NIF_TERM head;
    double value;

   this->data->clear(); 
   while (enif_get_list_cell(env, data, &head_row, &data)) 
    while (enif_get_list_cell(env, head_row, &head, &head_row))
        if (enif_get_double(env, head, &value)) {
            this->data->push_back(value);
        }
    

}


ERL_NIF_TERM PCudaMatrixFloatBuffer::toErlTerms(ErlNifEnv *env) {
    std::vector<double>::iterator iter;
    ERL_NIF_TERM retval = enif_make_list(env, 0);
    ERL_NIF_TERM row = enif_make_list(env, 0);

    if(this->rows() > 1){
        unsigned  j = 0;
            
        for (iter = this->data->end(); iter != this->data->begin();) {
            while( j < this->cols()){
                --iter;
                ++j;
                row = enif_make_list_cell(env, enif_make_double(env, *iter), row);
            }
            j = 0;
            retval = enif_make_list_cell(env, row, retval);
            row = enif_make_list(env, 0);
        }
    }else{
        if (this->data->size() > 0) {
            for (iter = this->data->end(); iter != this->data->begin();) {
                --iter;
                retval = enif_make_list_cell(env, enif_make_double(env, *iter), retval);
            }
        }        
    }
    return retval;
}



void PCudaMatrixFloatBuffer::clear() {
    this->data->clear();
}


//void PCudaMatrixFloatBuffer::mmul(PCudaBuffer *A, PCudaBuffer *B, PCudaBuffer *C, const int m, const int k, const int n){
void PCudaMatrixFloatBuffer::mmul(PCudaMatrixBuffer *A, PCudaMatrixBuffer *B, const int m, const int k, const int n){
    if (A->type() == BUF_TYPE_MATRIX_FLOAT && B->type() == BUF_TYPE_MATRIX_FLOAT) {
        PCudaMatrixFloatBuffer *fbA = (PCudaMatrixFloatBuffer *) A;
        PCudaMatrixFloatBuffer *fbB = (PCudaMatrixFloatBuffer *) B;
        //PCudaMatrixFloatBuffer *fbC = (PCudaMatrixFloatBuffer *) C;
        pcuda_mmul(fbA->data, fbB->data, this->data, m, k , n); 
    }
    
}
