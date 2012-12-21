-module(pteracuda_buffer).

-include("pteracuda_internals.hrl").

-export([new/1, new/4, new/5,
         destroy/1,
         size/1,
         write/2,
         read/1,
         duplicate/1,
         clear/1,
         sort/2,
         contains/3,
         intersection/3,
         minmax/2]).

-spec new(data_type) -> {ok, buffer()}.
new(integer) ->
    {ok, Buf} = pteracuda_nifs:new_int_buffer(),
    {ok, #pc_buffer{type = vector, data_type=integer, ref=Buf}};
new(float) ->
    {ok, Buf} = pteracuda_nifs:new_float_buffer(),
    {ok, #pc_buffer{type = vector, data_type=float, ref=Buf}};
new(string) ->
    {ok, Buf} = pteracuda_nifs:new_string_buffer(),
    {ok, #pc_buffer{type = vector, data_type=string, ref=Buf}}.

-spec new(matrix, data_type(), storage_layout(), matrix_rows(), matrix_columns()) -> {ok, buffer()}.
new(matrix, float, StorageLayout, Rows, Cols) ->
    {ok, Buf} = pteracuda_nifs:new_matrix_float_buffer(Rows,Cols),
    {ok, #pc_buffer{type = matrix, data_type=float, layout=StorageLayout, ref=Buf}};
new(matrix, integer, StorageLayout, Rows, Cols) ->
    {ok, Buf} = pteracuda_nifs:new_matrix_int_buffer(Rows,Cols),
    {ok, #pc_buffer{type = matrix, data_type=integer, layout=StorageLayout, ref=Buf}}.


-spec new(matrix, float, storage_layout(), float_matrix()) -> {ok, buffer()};
         (matrix, integer, storage_layout(), int_matrix()) -> {ok, buffer()}.
new(matrix, float, StorageLayout, Matrix) ->
    {ok, Buf} = pteracuda_nifs:new_matrix_float_buffer(Matrix),
    {ok, #pc_buffer{type = matrix, data_type=float, layout=StorageLayout,  ref=Buf}};    
new(matrix, integer, StorageLayout, Matrix) ->
    {ok, Buf} = pteracuda_nifs:new_matrix_int_buffer(Matrix),
    {ok, #pc_buffer{type = matrix, data_type=integer, layout=StorageLayout, ref=Buf}}.


destroy(#pc_buffer{ref=Ref}) ->
    pteracuda_nifs:destroy_buffer(Ref),
    ok.

size(#pc_buffer{ref=Ref}) ->
    pteracuda_nifs:buffer_size(Ref).

write(#pc_buffer{ref=Ref, data_type=Type}, Data) when Type =:= integer orelse
                                                 Type =:= string orelse
                                                 Type =:= float ->
    pteracuda_nifs:write_buffer(Ref, Data).

read(#pc_buffer{ref=Ref}) ->
    pteracuda_nifs:read_buffer(Ref).

duplicate(#pc_buffer{ref=Ref, data_type=Type}) when Type =:= integer orelse
                                               Type =:= string orelse
                                               Type =:= float ->
    {ok, OtherBuf} = new(Type),
    pteracuda_nifs:copy_buffer(Ref, OtherBuf#pc_buffer.ref),
    {ok, OtherBuf}.

clear(#pc_buffer{ref=Ref}) ->
    pteracuda_nifs:clear_buffer(Ref).

sort(#pc_context{ref=Ctx}, #pc_buffer{ref=Buf}) ->
    pteracuda_nifs:sort_buffer(Ctx, Buf).

contains(#pc_context{ref=Ctx}, #pc_buffer{ref=Buf}, Value) ->
    pteracuda_nifs:buffer_contains(Ctx, Buf, Value).

intersection(#pc_context{ref=Ctx}, #pc_buffer{ref=Buf1}, #pc_buffer{ref=Buf2}) ->
    pteracuda_nifs:buffer_intersection(Ctx, Buf1, Buf2).

minmax(#pc_context{ref=Ctx}, #pc_buffer{ref=Buf}) ->
    pteracuda_nifs:buffer_minmax(Ctx, Buf).
