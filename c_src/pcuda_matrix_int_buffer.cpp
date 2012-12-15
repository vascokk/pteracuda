#include <stdio.h>
#include "pcuda_buffer.h"
#include "pcuda_ops.h"

PCudaMatrixIntBuffer::PCudaMatrixIntBuffer() {
    this->data = new std::vector<long>();
    this->_rows = 1;
}

PCudaMatrixIntBuffer::PCudaMatrixIntBuffer(unsigned  rows, unsigned  cols) {
    this->_rows = rows;
    this->_cols = cols;
    this->data = new std::vector<long>(rows*cols);
}

PCudaMatrixIntBuffer::~PCudaMatrixIntBuffer() {
    delete this->data;
}

unsigned int PCudaMatrixIntBuffer::size() {
    return this->data->size();
}

void PCudaMatrixIntBuffer::write(ErlNifEnv *env, ERL_NIF_TERM data) {
    ERL_NIF_TERM head_row;
    ERL_NIF_TERM head;
    long value;

   this->data->clear();
   while (enif_get_list_cell(env, data, &head_row, &data)) 
    while (enif_get_list_cell(env, head_row, &head, &head_row))
        if (enif_get_long(env, head, &value)) {
            this->data->push_back(value);
        }
    

}


ERL_NIF_TERM PCudaMatrixIntBuffer::toErlTerms(ErlNifEnv *env) {
    std::vector<long>::iterator iter;
    ERL_NIF_TERM retval = enif_make_list(env, 0);
    ERL_NIF_TERM row = enif_make_list(env, 0);

    if(this->rows() > 1){
        unsigned  j = 0;
            
        for (iter = this->data->end(); iter != this->data->begin();) {
            while( j < this->cols()){
                --iter;
                ++j;
                row = enif_make_list_cell(env, enif_make_long(env, *iter), row);
            }
            j = 0;
            retval = enif_make_list_cell(env, row, retval);
            row = enif_make_list(env, 0);
        }
    }else{
        if (this->data->size() > 0) {
            for (iter = this->data->end(); iter != this->data->begin();) {
                --iter;
                retval = enif_make_list_cell(env, enif_make_long(env, *iter), retval);
            }
        }        
    }
    return retval;
}

void PCudaMatrixIntBuffer::clear() {
    this->data->clear();
}


/*void PCudaMatrixIntBuffer::mmul(PCudaMatrixBuffer *A, PCudaMatrixBuffer *B, const int m, const int k, const int n){
    if (A->type() == BUF_TYPE_MATRIX_INTEGER && B->type() == BUF_TYPE_MATRIX_INTEGER) {
        PCudaMatrixIntBuffer *fbA = (PCudaMatrixIntBuffer *) A;
        PCudaMatrixIntBuffer *fbB = (PCudaMatrixIntBuffer *) B;
        //PCudaMatrixIntBuffer *fbC = (PCudaMatrixIntBuffer *) C;
        pcuda_mmul(fbA->data, fbB->data, this->data, m, k , n); 
    }
    
}*/
