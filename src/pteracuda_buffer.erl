-module(pteracuda_buffer).

-include("include/pteracuda.hrl").

-export([new/1, new/2, new/4, new/5,
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

-spec new(data_type, integer()) -> {ok, buffer()}.
new(float, Size) ->
    {ok, Buf} = pteracuda_nifs:new_float_buffer(Size),
    %pteracuda_nifs:write_buffer(Buf, [0.0|| X<-lists:seq(1,Size)]),
    {ok, #pc_buffer{type = vector, data_type=float, ref=Buf}}.


-spec new(matrix, data_type(), orientation(), matrix_rows(), matrix_columns()) -> {ok, buffer()}.
new(matrix, float, Orientation, Rows, Cols) ->
    case Orientation of 
        row_major -> _orientation = ?ROW_MAJOR;
        column_major -> _orientation = ?COLUMN_MAJOR
    end,
    {ok, Buf} = pteracuda_nifs:new_matrix_float_buffer(Rows,Cols, _orientation),
    {ok, #pc_buffer{type = matrix, data_type=float, orientation=Orientation, ref=Buf}};
new(matrix, integer, Orientation, Rows, Cols) ->
    case Orientation of 
        row_major -> _orientation = ?ROW_MAJOR;
        column_major -> _orientation = ?COLUMN_MAJOR
    end,
    {ok, Buf} = pteracuda_nifs:new_matrix_int_buffer(Rows,Cols, _orientation),
    {ok, #pc_buffer{type = matrix, data_type=integer, orientation=Orientation, ref=Buf}}.


-spec new(matrix, float, orientation(), float_matrix()) -> {ok, buffer()};
         (matrix, integer, orientation(), int_matrix()) -> {ok, buffer()}.
new(matrix, float, Orientation, Matrix) ->
    case Orientation of 
        row_major -> _orientation = ?ROW_MAJOR;
        column_major -> _orientation = ?COLUMN_MAJOR
    end,
    {ok, Buf} = pteracuda_nifs:new_matrix_float_buffer(Matrix, _orientation),
    {ok, #pc_buffer{type = matrix, data_type=float, orientation=Orientation,  ref=Buf}};    
new(matrix, integer, Orientation, Matrix) ->
    case Orientation of 
        row_major -> _orientation = ?ROW_MAJOR;
        column_major -> _orientation = ?COLUMN_MAJOR
    end,
    {ok, Buf} = pteracuda_nifs:new_matrix_int_buffer(Matrix, _orientation),
    {ok, #pc_buffer{type = matrix, data_type=integer, orientation=Orientation, ref=Buf}}.


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
