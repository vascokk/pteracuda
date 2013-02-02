#ifndef PTERACUDA
#define PTERACUDA

#include "erl_nif.h"
#include "pcuda_buffer.h"
#include "cuda.h"


extern "C" {
    static int pteracuda_on_load(ErlNifEnv *env, void **priv_data, ERL_NIF_TERM load_info);

    ERL_NIF_TERM pteracuda_nifs_new_context(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
    ERL_NIF_TERM pteracuda_nifs_destroy_context(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);

    ERL_NIF_TERM pteracuda_nifs_new_int_buffer(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
    ERL_NIF_TERM pteracuda_nifs_new_string_buffer(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
    ERL_NIF_TERM pteracuda_nifs_new_float_buffer(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);

    ERL_NIF_TERM pteracuda_nifs_destroy_buffer(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
    ERL_NIF_TERM pteracuda_nifs_buffer_size(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);

    ERL_NIF_TERM pteracuda_nifs_write_buffer(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
    ERL_NIF_TERM pteracuda_nifs_read_buffer(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
    ERL_NIF_TERM pteracuda_nifs_buffer_delete(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
    ERL_NIF_TERM pteracuda_nifs_buffer_insert(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
    ERL_NIF_TERM pteracuda_nifs_sort_buffer(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
    ERL_NIF_TERM pteracuda_nifs_clear_buffer(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
    ERL_NIF_TERM pteracuda_nifs_buffer_contains(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
    ERL_NIF_TERM pteracuda_nifs_copy_buffer(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
    ERL_NIF_TERM pteracuda_nifs_buffer_intersection(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
    ERL_NIF_TERM pteracuda_nifs_buffer_minmax(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);

    ERL_NIF_TERM pteracuda_nifs_new_matrix_int_buffer(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
    ERL_NIF_TERM pteracuda_nifs_new_matrix_float_buffer(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
    
    ERL_NIF_TERM pteracuda_nifs_gemm(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
    ERL_NIF_TERM pteracuda_nifs_gemv(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
    ERL_NIF_TERM pteracuda_nifs_saxpy(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
    ERL_NIF_TERM pteracuda_nifs_transpose(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
    ERL_NIF_TERM pteracuda_nifs_geam(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
    ERL_NIF_TERM pteracuda_nifs_smm(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);

    ERL_NIF_TERM pteracuda_nifs_sigmoid(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
    ERL_NIF_TERM pteracuda_nifs_tanh(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
    ERL_NIF_TERM pteracuda_nifs_log(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);

    //ML functions
    ERL_NIF_TERM pteracuda_ml_gd(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
    ERL_NIF_TERM pteracuda_ml_gd_learn(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);


    static ErlNifFunc pteracuda_nif_funcs[] = {
        {"new_context", 0, pteracuda_nifs_new_context},
        {"new_context", 1, pteracuda_nifs_new_context},
        {"destroy_context", 1, pteracuda_nifs_destroy_context},
        {"new_int_buffer", 0, pteracuda_nifs_new_int_buffer},
        {"new_int_buffer", 1, pteracuda_nifs_new_int_buffer},
        {"new_string_buffer", 0, pteracuda_nifs_new_string_buffer},
        {"new_float_buffer", 0, pteracuda_nifs_new_float_buffer},
        {"new_float_buffer", 1, pteracuda_nifs_new_float_buffer},
        {"destroy_buffer", 1, pteracuda_nifs_destroy_buffer},
        {"buffer_size", 1, pteracuda_nifs_buffer_size},
        {"write_buffer", 2, pteracuda_nifs_write_buffer},
        {"buffer_delete", 2, pteracuda_nifs_buffer_delete},
        {"buffer_insert", 3, pteracuda_nifs_buffer_insert},
        {"read_buffer", 1, pteracuda_nifs_read_buffer},
        {"sort_buffer", 2, pteracuda_nifs_sort_buffer},
        {"clear_buffer", 1, pteracuda_nifs_clear_buffer},
        {"buffer_contains", 3, pteracuda_nifs_buffer_contains},
        {"copy_buffer", 2, pteracuda_nifs_copy_buffer},
        {"buffer_intersection", 3, pteracuda_nifs_buffer_intersection},
        {"buffer_minmax", 2, pteracuda_nifs_buffer_minmax},

        {"new_matrix_int_buffer", 2, pteracuda_nifs_new_matrix_int_buffer},
        {"new_matrix_float_buffer", 2, pteracuda_nifs_new_matrix_float_buffer},
        {"new_matrix_int_buffer", 3, pteracuda_nifs_new_matrix_int_buffer},
        {"new_matrix_float_buffer", 3, pteracuda_nifs_new_matrix_float_buffer},

        {"gemm", 11, pteracuda_nifs_gemm},
        {"gemv", 9, pteracuda_nifs_gemv},
        {"saxpy", 4, pteracuda_nifs_saxpy},
        {"transpose", 3, pteracuda_nifs_transpose},
        {"geam", 10, pteracuda_nifs_geam},
        {"smm", 4, pteracuda_nifs_smm},
        {"sigmoid", 3, pteracuda_nifs_sigmoid},
        {"tanh", 3, pteracuda_nifs_tanh},
        {"log", 3, pteracuda_nifs_log},

        //ML functions
        {"gd", 6, pteracuda_ml_gd},
        {"gd_learn", 8, pteracuda_ml_gd_learn}      

    };
};

struct PCudaBufferRef {
    PCudaBuffer *buffer;
    bool destroyed;
};

struct PCudaContextRef {
    CUcontext ctx;
    bool destroyed;
};

extern ErlNifResourceType *pteracuda_buffer_resource;
extern ErlNifResourceType *pteracuda_context_resource;

extern ERL_NIF_TERM ATOM_TRUE;
extern ERL_NIF_TERM ATOM_FALSE;
extern ERL_NIF_TERM ATOM_OK;
extern ERL_NIF_TERM ATOM_ERROR;
extern ERL_NIF_TERM ATOM_WRONG_TYPE;
extern ERL_NIF_TERM OOM_ERROR;


#endif
