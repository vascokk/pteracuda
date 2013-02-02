#include "pteracuda.h"
#include "pcuda_buffer.h"
#include "pcuda_kernels.h"

#include <stdio.h>
#include <iostream>

#include "pcuda_ml.h"


ERL_NIF_TERM pteracuda_ml_gd(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    PCudaContextRef *ctxRef;
    PCudaBufferRef *ref_Theta, *ref_X, *ref_Y;
    unsigned long num_features, num_samples;
    
    if (argc != 6 || !enif_get_resource(env, argv[0], pteracuda_context_resource, (void **) &ctxRef) ||
        !enif_get_resource(env, argv[1], pteracuda_buffer_resource, (void **) &ref_Theta) ||
        !enif_get_resource(env, argv[2], pteracuda_buffer_resource, (void **) &ref_X) ||
        !enif_get_resource(env, argv[3], pteracuda_buffer_resource, (void **) &ref_Y) ||
        !enif_get_ulong(env, argv[4], &num_features) ||
        !enif_get_ulong(env, argv[5], &num_samples)

        ) {
        return enif_make_badarg(env);
    }

    cuCtxSetCurrent(ctxRef->ctx);

    pcuda_gd(((PCudaFloatBuffer*)ref_Theta->buffer)->get_data(), ((PCudaFloatBuffer*)ref_X->buffer)->get_data(), ((PCudaFloatBuffer*)ref_Y->buffer)->get_data(), num_features, num_samples);
    
    return ATOM_OK;
}

ERL_NIF_TERM pteracuda_ml_gd_learn(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    PCudaContextRef *ctxRef;
    PCudaBufferRef *ref_Theta, *ref_X, *ref_Y;
    unsigned long num_features; 
    unsigned long num_samples;
    unsigned long iterations;
    double learning_rate;
    
    if (argc != 8 || !enif_get_resource(env, argv[0], pteracuda_context_resource, (void **) &ctxRef) ||
        !enif_get_resource(env, argv[1], pteracuda_buffer_resource, (void **) &ref_Theta) ||
        !enif_get_resource(env, argv[2], pteracuda_buffer_resource, (void **) &ref_X) ||
        !enif_get_resource(env, argv[3], pteracuda_buffer_resource, (void **) &ref_Y) ||
        !enif_get_ulong(env, argv[4], &num_features) ||
        !enif_get_ulong(env, argv[5], &num_samples) ||
        !enif_get_double(env, argv[6], &learning_rate) ||
        !enif_get_ulong(env, argv[7], &iterations)

        ) {
        return enif_make_badarg(env);
    }

    cuCtxSetCurrent(ctxRef->ctx);

    pcuda_gd_learn(((PCudaFloatBuffer*)ref_Theta->buffer)->get_data(), ((PCudaFloatBuffer*)ref_X->buffer)->get_data(), ((PCudaFloatBuffer*)ref_Y->buffer)->get_data(), num_features, num_samples, (float)learning_rate, iterations);
    
    return ATOM_OK;
}