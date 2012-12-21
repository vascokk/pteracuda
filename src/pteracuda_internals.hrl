-define (NO_TRANSPOSE, 0).
-define (TRANSPOSE, 1).
-define (CONJUGATE_TRANSPOSE, 2).


-type int_vector() :: [integer()].
-type float_vector() :: [float()].
-type int_matrix() :: [[integer()]].
-type float_matrix() :: [[float()]].
-type matrix() :: int_matrix() | float_matrix().
-type matrix_rows() :: integer().
-type matrix_columns() :: integer().
-type storage_type() :: vector | matrix.
-type data_type() :: integer | float | string.
-type storage_layout() :: row_major | column_major.


-record(pc_buffer, {type      :: storage_type(),
					data_type :: data_type(),
					layout    :: storage_layout(),
                    ref       :: term()}).

-record(pc_context, {ref}).

-type buffer() :: #pc_buffer{}.

