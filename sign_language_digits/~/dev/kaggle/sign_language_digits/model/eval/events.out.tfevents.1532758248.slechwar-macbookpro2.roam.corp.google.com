       �K"	   :��Abrain.Event:2gV�+�w     P�I	+:��A"��

global_step/Initializer/zerosConst*
_class
loc:@global_step*
value	B	 R *
dtype0	*
_output_shapes
: 
�
global_step
VariableV2*
	container *
shape: *
dtype0	*
_output_shapes
: *
shared_name *
_class
loc:@global_step
�
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
T0	*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: *
use_locking(
j
global_step/readIdentityglobal_step*
T0	*
_class
loc:@global_step*
_output_shapes
: 
�
enqueue_input/fifo_queueFIFOQueueV2"/device:CPU:0*
shapes
: :� :*
shared_name *
capacity�*
	container *
_output_shapes
: *
component_types
2	
m
enqueue_input/PlaceholderPlaceholder"/device:CPU:0*
dtype0	*
_output_shapes
:*
shape:
o
enqueue_input/Placeholder_1Placeholder"/device:CPU:0*
dtype0*
_output_shapes
:*
shape:
o
enqueue_input/Placeholder_2Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:
�
$enqueue_input/fifo_queue_EnqueueManyQueueEnqueueManyV2enqueue_input/fifo_queueenqueue_input/Placeholderenqueue_input/Placeholder_1enqueue_input/Placeholder_2"/device:CPU:0*
Tcomponents
2	*

timeout_ms���������
v
enqueue_input/fifo_queue_CloseQueueCloseV2enqueue_input/fifo_queue"/device:CPU:0*
cancel_pending_enqueues( 
x
 enqueue_input/fifo_queue_Close_1QueueCloseV2enqueue_input/fifo_queue"/device:CPU:0*
cancel_pending_enqueues(
m
enqueue_input/fifo_queue_SizeQueueSizeV2enqueue_input/fifo_queue"/device:CPU:0*
_output_shapes
: 
d
enqueue_input/sub/yConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
|
enqueue_input/subSubenqueue_input/fifo_queue_Sizeenqueue_input/sub/y"/device:CPU:0*
T0*
_output_shapes
: 
h
enqueue_input/Maximum/xConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
|
enqueue_input/MaximumMaximumenqueue_input/Maximum/xenqueue_input/sub"/device:CPU:0*
T0*
_output_shapes
: 
p
enqueue_input/CastCastenqueue_input/Maximum"/device:CPU:0*
_output_shapes
: *

DstT0*

SrcT0
g
enqueue_input/mul/yConst"/device:CPU:0*
valueB
 *o�:*
dtype0*
_output_shapes
: 
q
enqueue_input/mulMulenqueue_input/Castenqueue_input/mul/y"/device:CPU:0*
T0*
_output_shapes
: 
�
Menqueue_input/queue/enqueue_input/fifo_queuefraction_over_0_of_1000_full/tagsConst"/device:CPU:0*
_output_shapes
: *Y
valuePBN BHenqueue_input/queue/enqueue_input/fifo_queuefraction_over_0_of_1000_full*
dtype0
�
Henqueue_input/queue/enqueue_input/fifo_queuefraction_over_0_of_1000_fullScalarSummaryMenqueue_input/queue/enqueue_input/fifo_queuefraction_over_0_of_1000_full/tagsenqueue_input/mul"/device:CPU:0*
T0*
_output_shapes
: 
i
fifo_queue_DequeueUpTo/nConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
fifo_queue_DequeueUpToQueueDequeueUpToV2enqueue_input/fifo_queuefifo_queue_DequeueUpTo/n"/device:CPU:0*J
_output_shapes8
6:���������:���������� :���������*
component_types
2	*

timeout_ms���������
�
2dnn/input_from_feature_columns/input_layer/x/ShapeShapefifo_queue_DequeueUpTo:1*
_output_shapes
:*
T0*
out_type0
�
@dnn/input_from_feature_columns/input_layer/x/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
Bdnn/input_from_feature_columns/input_layer/x/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
Bdnn/input_from_feature_columns/input_layer/x/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
�
:dnn/input_from_feature_columns/input_layer/x/strided_sliceStridedSlice2dnn/input_from_feature_columns/input_layer/x/Shape@dnn/input_from_feature_columns/input_layer/x/strided_slice/stackBdnn/input_from_feature_columns/input_layer/x/strided_slice/stack_1Bdnn/input_from_feature_columns/input_layer/x/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask

<dnn/input_from_feature_columns/input_layer/x/Reshape/shape/1Const*
value
B :� *
dtype0*
_output_shapes
: 
�
:dnn/input_from_feature_columns/input_layer/x/Reshape/shapePack:dnn/input_from_feature_columns/input_layer/x/strided_slice<dnn/input_from_feature_columns/input_layer/x/Reshape/shape/1*

axis *
N*
_output_shapes
:*
T0
�
4dnn/input_from_feature_columns/input_layer/x/ReshapeReshapefifo_queue_DequeueUpTo:1:dnn/input_from_feature_columns/input_layer/x/Reshape/shape*(
_output_shapes
:���������� *
T0*
Tshape0
~
<dnn/input_from_feature_columns/input_layer/concat/concat_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
1dnn/input_from_feature_columns/input_layer/concatIdentity4dnn/input_from_feature_columns/input_layer/x/Reshape*(
_output_shapes
:���������� *
T0
�
@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shapeConst*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
valueB"      *
dtype0*
_output_shapes
:
�
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/minConst*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
valueB
 *0�*
dtype0*
_output_shapes
: 
�
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/maxConst*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
valueB
 *0=*
dtype0*
_output_shapes
: 
�
Hdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shape*

seed *
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
seed2 *
dtype0* 
_output_shapes
:
� �
�
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*
_output_shapes
: *
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0
�
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0* 
_output_shapes
:
� �
�
:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0* 
_output_shapes
:
� �
�
dnn/hiddenlayer_0/kernel/part_0
VariableV2*
shared_name *2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
	container *
shape:
� �*
dtype0* 
_output_shapes
:
� �
�
&dnn/hiddenlayer_0/kernel/part_0/AssignAssigndnn/hiddenlayer_0/kernel/part_0:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
validate_shape(* 
_output_shapes
:
� �*
use_locking(*
T0
�
$dnn/hiddenlayer_0/kernel/part_0/readIdentitydnn/hiddenlayer_0/kernel/part_0*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0* 
_output_shapes
:
� �
�
/dnn/hiddenlayer_0/bias/part_0/Initializer/zerosConst*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
valueB�*    *
dtype0*
_output_shapes	
:�
�
dnn/hiddenlayer_0/bias/part_0
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0
�
$dnn/hiddenlayer_0/bias/part_0/AssignAssigndnn/hiddenlayer_0/bias/part_0/dnn/hiddenlayer_0/bias/part_0/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
validate_shape(
�
"dnn/hiddenlayer_0/bias/part_0/readIdentitydnn/hiddenlayer_0/bias/part_0*
_output_shapes	
:�*
T0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0
u
dnn/hiddenlayer_0/kernelIdentity$dnn/hiddenlayer_0/kernel/part_0/read*
T0* 
_output_shapes
:
� �
�
dnn/hiddenlayer_0/MatMulMatMul1dnn/input_from_feature_columns/input_layer/concatdnn/hiddenlayer_0/kernel*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
l
dnn/hiddenlayer_0/biasIdentity"dnn/hiddenlayer_0/bias/part_0/read*
_output_shapes	
:�*
T0
�
dnn/hiddenlayer_0/BiasAddBiasAdddnn/hiddenlayer_0/MatMuldnn/hiddenlayer_0/bias*
data_formatNHWC*(
_output_shapes
:����������*
T0
l
dnn/hiddenlayer_0/ReluReludnn/hiddenlayer_0/BiasAdd*
T0*(
_output_shapes
:����������
[
dnn/zero_fraction/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
dnn/zero_fraction/EqualEqualdnn/hiddenlayer_0/Reludnn/zero_fraction/zero*(
_output_shapes
:����������*
T0
y
dnn/zero_fraction/CastCastdnn/zero_fraction/Equal*(
_output_shapes
:����������*

DstT0*

SrcT0

h
dnn/zero_fraction/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
dnn/zero_fraction/MeanMeandnn/zero_fraction/Castdnn/zero_fraction/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsConst*>
value5B3 B-dnn/dnn/hiddenlayer_0/fraction_of_zero_values*
dtype0*
_output_shapes
: 
�
-dnn/dnn/hiddenlayer_0/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsdnn/zero_fraction/Mean*
T0*
_output_shapes
: 
�
$dnn/dnn/hiddenlayer_0/activation/tagConst*1
value(B& B dnn/dnn/hiddenlayer_0/activation*
dtype0*
_output_shapes
: 
�
 dnn/dnn/hiddenlayer_0/activationHistogramSummary$dnn/dnn/hiddenlayer_0/activation/tagdnn/hiddenlayer_0/Relu*
T0*
_output_shapes
: 
�
@dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shapeConst*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
valueB"       *
dtype0*
_output_shapes
:
�
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/minConst*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
valueB
 *:��*
dtype0*
_output_shapes
: 
�
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/maxConst*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
valueB
 *:�>*
dtype0*
_output_shapes
: 
�
Hdnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	� *

seed *
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
seed2 
�
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes
: 
�
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes
:	� 
�
:dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes
:	� *
T0
�
dnn/hiddenlayer_1/kernel/part_0
VariableV2*
	container *
shape:	� *
dtype0*
_output_shapes
:	� *
shared_name *2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0
�
&dnn/hiddenlayer_1/kernel/part_0/AssignAssigndnn/hiddenlayer_1/kernel/part_0:dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform*
use_locking(*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
validate_shape(*
_output_shapes
:	� 
�
$dnn/hiddenlayer_1/kernel/part_0/readIdentitydnn/hiddenlayer_1/kernel/part_0*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes
:	� 
�
/dnn/hiddenlayer_1/bias/part_0/Initializer/zerosConst*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
valueB *    *
dtype0*
_output_shapes
: 
�
dnn/hiddenlayer_1/bias/part_0
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
	container 
�
$dnn/hiddenlayer_1/bias/part_0/AssignAssigndnn/hiddenlayer_1/bias/part_0/dnn/hiddenlayer_1/bias/part_0/Initializer/zeros*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
"dnn/hiddenlayer_1/bias/part_0/readIdentitydnn/hiddenlayer_1/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
_output_shapes
: *
T0
t
dnn/hiddenlayer_1/kernelIdentity$dnn/hiddenlayer_1/kernel/part_0/read*
T0*
_output_shapes
:	� 
�
dnn/hiddenlayer_1/MatMulMatMuldnn/hiddenlayer_0/Reludnn/hiddenlayer_1/kernel*'
_output_shapes
:��������� *
transpose_a( *
transpose_b( *
T0
k
dnn/hiddenlayer_1/biasIdentity"dnn/hiddenlayer_1/bias/part_0/read*
T0*
_output_shapes
: 
�
dnn/hiddenlayer_1/BiasAddBiasAdddnn/hiddenlayer_1/MatMuldnn/hiddenlayer_1/bias*
data_formatNHWC*'
_output_shapes
:��������� *
T0
k
dnn/hiddenlayer_1/ReluReludnn/hiddenlayer_1/BiasAdd*
T0*'
_output_shapes
:��������� 
]
dnn/zero_fraction_1/zeroConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
dnn/zero_fraction_1/EqualEqualdnn/hiddenlayer_1/Reludnn/zero_fraction_1/zero*
T0*'
_output_shapes
:��������� 
|
dnn/zero_fraction_1/CastCastdnn/zero_fraction_1/Equal*

SrcT0
*'
_output_shapes
:��������� *

DstT0
j
dnn/zero_fraction_1/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
dnn/zero_fraction_1/MeanMeandnn/zero_fraction_1/Castdnn/zero_fraction_1/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
2dnn/dnn/hiddenlayer_1/fraction_of_zero_values/tagsConst*>
value5B3 B-dnn/dnn/hiddenlayer_1/fraction_of_zero_values*
dtype0*
_output_shapes
: 
�
-dnn/dnn/hiddenlayer_1/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_1/fraction_of_zero_values/tagsdnn/zero_fraction_1/Mean*
_output_shapes
: *
T0
�
$dnn/dnn/hiddenlayer_1/activation/tagConst*1
value(B& B dnn/dnn/hiddenlayer_1/activation*
dtype0*
_output_shapes
: 
�
 dnn/dnn/hiddenlayer_1/activationHistogramSummary$dnn/dnn/hiddenlayer_1/activation/tagdnn/hiddenlayer_1/Relu*
T0*
_output_shapes
: 
�
9dnn/logits/kernel/part_0/Initializer/random_uniform/shapeConst*+
_class!
loc:@dnn/logits/kernel/part_0*
valueB"       *
dtype0*
_output_shapes
:
�
7dnn/logits/kernel/part_0/Initializer/random_uniform/minConst*
_output_shapes
: *+
_class!
loc:@dnn/logits/kernel/part_0*
valueB
 *JQھ*
dtype0
�
7dnn/logits/kernel/part_0/Initializer/random_uniform/maxConst*+
_class!
loc:@dnn/logits/kernel/part_0*
valueB
 *JQ�>*
dtype0*
_output_shapes
: 
�
Adnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform9dnn/logits/kernel/part_0/Initializer/random_uniform/shape*
dtype0*
_output_shapes

: *

seed *
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
seed2 
�
7dnn/logits/kernel/part_0/Initializer/random_uniform/subSub7dnn/logits/kernel/part_0/Initializer/random_uniform/max7dnn/logits/kernel/part_0/Initializer/random_uniform/min*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes
: *
T0
�
7dnn/logits/kernel/part_0/Initializer/random_uniform/mulMulAdnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniform7dnn/logits/kernel/part_0/Initializer/random_uniform/sub*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

: 
�
3dnn/logits/kernel/part_0/Initializer/random_uniformAdd7dnn/logits/kernel/part_0/Initializer/random_uniform/mul7dnn/logits/kernel/part_0/Initializer/random_uniform/min*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

: *
T0
�
dnn/logits/kernel/part_0
VariableV2*
shared_name *+
_class!
loc:@dnn/logits/kernel/part_0*
	container *
shape
: *
dtype0*
_output_shapes

: 
�
dnn/logits/kernel/part_0/AssignAssigndnn/logits/kernel/part_03dnn/logits/kernel/part_0/Initializer/random_uniform*
validate_shape(*
_output_shapes

: *
use_locking(*
T0*+
_class!
loc:@dnn/logits/kernel/part_0
�
dnn/logits/kernel/part_0/readIdentitydnn/logits/kernel/part_0*
_output_shapes

: *
T0*+
_class!
loc:@dnn/logits/kernel/part_0
�
(dnn/logits/bias/part_0/Initializer/zerosConst*)
_class
loc:@dnn/logits/bias/part_0*
valueB*    *
dtype0*
_output_shapes
:
�
dnn/logits/bias/part_0
VariableV2*
_output_shapes
:*
shared_name *)
_class
loc:@dnn/logits/bias/part_0*
	container *
shape:*
dtype0
�
dnn/logits/bias/part_0/AssignAssigndnn/logits/bias/part_0(dnn/logits/bias/part_0/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@dnn/logits/bias/part_0*
validate_shape(*
_output_shapes
:
�
dnn/logits/bias/part_0/readIdentitydnn/logits/bias/part_0*)
_class
loc:@dnn/logits/bias/part_0*
_output_shapes
:*
T0
e
dnn/logits/kernelIdentitydnn/logits/kernel/part_0/read*
_output_shapes

: *
T0
�
dnn/logits/MatMulMatMuldnn/hiddenlayer_1/Reludnn/logits/kernel*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
]
dnn/logits/biasIdentitydnn/logits/bias/part_0/read*
T0*
_output_shapes
:
�
dnn/logits/BiasAddBiasAdddnn/logits/MatMuldnn/logits/bias*
T0*
data_formatNHWC*'
_output_shapes
:���������
]
dnn/zero_fraction_2/zeroConst*
_output_shapes
: *
valueB
 *    *
dtype0
�
dnn/zero_fraction_2/EqualEqualdnn/logits/BiasAdddnn/zero_fraction_2/zero*'
_output_shapes
:���������*
T0
|
dnn/zero_fraction_2/CastCastdnn/zero_fraction_2/Equal*

SrcT0
*'
_output_shapes
:���������*

DstT0
j
dnn/zero_fraction_2/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
dnn/zero_fraction_2/MeanMeandnn/zero_fraction_2/Castdnn/zero_fraction_2/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
+dnn/dnn/logits/fraction_of_zero_values/tagsConst*7
value.B, B&dnn/dnn/logits/fraction_of_zero_values*
dtype0*
_output_shapes
: 
�
&dnn/dnn/logits/fraction_of_zero_valuesScalarSummary+dnn/dnn/logits/fraction_of_zero_values/tagsdnn/zero_fraction_2/Mean*
T0*
_output_shapes
: 
w
dnn/dnn/logits/activation/tagConst**
value!B Bdnn/dnn/logits/activation*
dtype0*
_output_shapes
: 
�
dnn/dnn/logits/activationHistogramSummarydnn/dnn/logits/activation/tagdnn/logits/BiasAdd*
T0*
_output_shapes
: 
s
!dnn/head/predictions/logits/ShapeShapednn/logits/BiasAdd*
out_type0*
_output_shapes
:*
T0
w
5dnn/head/predictions/logits/assert_rank_at_least/rankConst*
value	B :*
dtype0*
_output_shapes
: 
g
_dnn/head/predictions/logits/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp
X
Pdnn/head/predictions/logits/assert_rank_at_least/static_checks_determined_all_okNoOp
n
dnn/head/predictions/logisticSigmoiddnn/logits/BiasAdd*
T0*'
_output_shapes
:���������
r
dnn/head/predictions/zeros_like	ZerosLikednn/logits/BiasAdd*
T0*'
_output_shapes
:���������
u
*dnn/head/predictions/two_class_logits/axisConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
%dnn/head/predictions/two_class_logitsConcatV2dnn/head/predictions/zeros_likednn/logits/BiasAdd*dnn/head/predictions/two_class_logits/axis*
T0*
N*'
_output_shapes
:���������*

Tidx0
�
"dnn/head/predictions/probabilitiesSoftmax%dnn/head/predictions/two_class_logits*
T0*'
_output_shapes
:���������
s
(dnn/head/predictions/class_ids/dimensionConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
dnn/head/predictions/class_idsArgMax%dnn/head/predictions/two_class_logits(dnn/head/predictions/class_ids/dimension*
T0*
output_type0	*#
_output_shapes
:���������*

Tidx0
n
#dnn/head/predictions/ExpandDims/dimConst*
_output_shapes
: *
valueB :
���������*
dtype0
�
dnn/head/predictions/ExpandDims
ExpandDimsdnn/head/predictions/class_ids#dnn/head/predictions/ExpandDims/dim*

Tdim0*
T0	*'
_output_shapes
:���������
�
 dnn/head/predictions/str_classesAsStringdnn/head/predictions/ExpandDims*
T0	*

fill *

scientific( *
width���������*'
_output_shapes
:���������*
	precision���������*
shortest( 
m
dnn/head/labels/ShapeShapefifo_queue_DequeueUpTo:2*
T0*
out_type0*
_output_shapes
:
i
dnn/head/labels/Shape_1Shapednn/logits/BiasAdd*
T0*
out_type0*
_output_shapes
:
k
)dnn/head/labels/assert_rank_at_least/rankConst*
value	B :*
dtype0*
_output_shapes
: 
�
*dnn/head/labels/assert_rank_at_least/ShapeShapefifo_queue_DequeueUpTo:2*
T0*
out_type0*
_output_shapes
:
[
Sdnn/head/labels/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp
L
Ddnn/head/labels/assert_rank_at_least/static_checks_determined_all_okNoOp
�
#dnn/head/labels/strided_slice/stackConstE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok*
dtype0*
_output_shapes
:*
valueB: 
�
%dnn/head/labels/strided_slice/stack_1ConstE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok*
valueB:
���������*
dtype0*
_output_shapes
:
�
%dnn/head/labels/strided_slice/stack_2ConstE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok*
valueB:*
dtype0*
_output_shapes
:
�
dnn/head/labels/strided_sliceStridedSlicednn/head/labels/Shape_1#dnn/head/labels/strided_slice/stack%dnn/head/labels/strided_slice/stack_1%dnn/head/labels/strided_slice/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:*
T0*
Index0
�
dnn/head/labels/concat/values_1ConstE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok*
valueB:*
dtype0*
_output_shapes
:
�
dnn/head/labels/concat/axisConstE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok*
dtype0*
_output_shapes
: *
value	B : 
�
dnn/head/labels/concatConcatV2dnn/head/labels/strided_slicednn/head/labels/concat/values_1dnn/head/labels/concat/axis*
_output_shapes
:*

Tidx0*
T0*
N

"dnn/head/labels/assert_equal/EqualEqualdnn/head/labels/concatdnn/head/labels/Shape*
_output_shapes
:*
T0
�
"dnn/head/labels/assert_equal/ConstConstE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok*
valueB: *
dtype0*
_output_shapes
:
�
 dnn/head/labels/assert_equal/AllAll"dnn/head/labels/assert_equal/Equal"dnn/head/labels/assert_equal/Const*
	keep_dims( *

Tidx0*
_output_shapes
: 
�
)dnn/head/labels/assert_equal/Assert/ConstConstE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok*(
valueB Bexpected_labels_shape: *
dtype0*
_output_shapes
: 
�
+dnn/head/labels/assert_equal/Assert/Const_1ConstE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok*
valueB Blabels_shape: *
dtype0*
_output_shapes
: 
�
1dnn/head/labels/assert_equal/Assert/Assert/data_0ConstE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok*(
valueB Bexpected_labels_shape: *
dtype0*
_output_shapes
: 
�
1dnn/head/labels/assert_equal/Assert/Assert/data_2ConstE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok*
valueB Blabels_shape: *
dtype0*
_output_shapes
: 
�
*dnn/head/labels/assert_equal/Assert/AssertAssert dnn/head/labels/assert_equal/All1dnn/head/labels/assert_equal/Assert/Assert/data_0dnn/head/labels/concat1dnn/head/labels/assert_equal/Assert/Assert/data_2dnn/head/labels/Shape*
T
2*
	summarize
�
dnn/head/labelsIdentityfifo_queue_DequeueUpTo:2+^dnn/head/labels/assert_equal/Assert/AssertE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok*
T0*'
_output_shapes
:���������
j
dnn/head/ToFloatCastdnn/head/labels*

SrcT0*'
_output_shapes
:���������*

DstT0
`
dnn/head/assert_range/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *   @
�
&dnn/head/assert_range/assert_less/LessLessdnn/head/ToFloatdnn/head/assert_range/Const*
T0*'
_output_shapes
:���������
x
'dnn/head/assert_range/assert_less/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
%dnn/head/assert_range/assert_less/AllAll&dnn/head/assert_range/assert_less/Less'dnn/head/assert_range/assert_less/Const*
	keep_dims( *

Tidx0*
_output_shapes
: 
�
.dnn/head/assert_range/assert_less/Assert/ConstConst*+
value"B  BLabel IDs must < n_classes*
dtype0*
_output_shapes
: 
�
0dnn/head/assert_range/assert_less/Assert/Const_1Const*;
value2B0 B*Condition x < y did not hold element-wise:*
dtype0*
_output_shapes
: 
�
0dnn/head/assert_range/assert_less/Assert/Const_2Const**
value!B Bx (dnn/head/ToFloat:0) = *
dtype0*
_output_shapes
: 
�
0dnn/head/assert_range/assert_less/Assert/Const_3Const*5
value,B* B$y (dnn/head/assert_range/Const:0) = *
dtype0*
_output_shapes
: 
�
;dnn/head/assert_range/assert_less/Assert/AssertGuard/SwitchSwitch%dnn/head/assert_range/assert_less/All%dnn/head/assert_range/assert_less/All*
T0
*
_output_shapes
: : 
�
=dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_tIdentity=dnn/head/assert_range/assert_less/Assert/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
�
=dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_fIdentity;dnn/head/assert_range/assert_less/Assert/AssertGuard/Switch*
_output_shapes
: *
T0

�
<dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_idIdentity%dnn/head/assert_range/assert_less/All*
T0
*
_output_shapes
: 
�
9dnn/head/assert_range/assert_less/Assert/AssertGuard/NoOpNoOp>^dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_t
�
Gdnn/head/assert_range/assert_less/Assert/AssertGuard/control_dependencyIdentity=dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_t:^dnn/head/assert_range/assert_less/Assert/AssertGuard/NoOp*
_output_shapes
: *
T0
*P
_classF
DBloc:@dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_t
�
Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_0Const>^dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f*+
value"B  BLabel IDs must < n_classes*
dtype0*
_output_shapes
: 
�
Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_1Const>^dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f*
dtype0*
_output_shapes
: *;
value2B0 B*Condition x < y did not hold element-wise:
�
Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_2Const>^dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f**
value!B Bx (dnn/head/ToFloat:0) = *
dtype0*
_output_shapes
: 
�
Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_4Const>^dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f*5
value,B* B$y (dnn/head/assert_range/Const:0) = *
dtype0*
_output_shapes
: 
�
;dnn/head/assert_range/assert_less/Assert/AssertGuard/AssertAssertBdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/SwitchBdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_0Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_1Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_2Ddnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/Switch_1Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_4Ddnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/Switch_2*
T

2*
	summarize
�
Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/SwitchSwitch%dnn/head/assert_range/assert_less/All<dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_id*
T0
*8
_class.
,*loc:@dnn/head/assert_range/assert_less/All*
_output_shapes
: : 
�
Ddnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/Switch_1Switchdnn/head/ToFloat<dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_id*
T0*#
_class
loc:@dnn/head/ToFloat*:
_output_shapes(
&:���������:���������
�
Ddnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/Switch_2Switchdnn/head/assert_range/Const<dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_id*
_output_shapes
: : *
T0*.
_class$
" loc:@dnn/head/assert_range/Const
�
Idnn/head/assert_range/assert_less/Assert/AssertGuard/control_dependency_1Identity=dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f<^dnn/head/assert_range/assert_less/Assert/AssertGuard/Assert*
_output_shapes
: *
T0
*P
_classF
DBloc:@dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f
�
:dnn/head/assert_range/assert_less/Assert/AssertGuard/MergeMergeIdnn/head/assert_range/assert_less/Assert/AssertGuard/control_dependency_1Gdnn/head/assert_range/assert_less/Assert/AssertGuard/control_dependency*
_output_shapes
: : *
T0
*
N
t
/dnn/head/assert_range/assert_non_negative/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Ednn/head/assert_range/assert_non_negative/assert_less_equal/LessEqual	LessEqual/dnn/head/assert_range/assert_non_negative/Constdnn/head/ToFloat*
T0*'
_output_shapes
:���������
�
Adnn/head/assert_range/assert_non_negative/assert_less_equal/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
?dnn/head/assert_range/assert_non_negative/assert_less_equal/AllAllEdnn/head/assert_range/assert_non_negative/assert_less_equal/LessEqualAdnn/head/assert_range/assert_non_negative/assert_less_equal/Const*
	keep_dims( *

Tidx0*
_output_shapes
: 
�
Hdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/ConstConst*$
valueB BLabel IDs must >= 0*
dtype0*
_output_shapes
: 
�
Jdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/Const_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+Condition x >= 0 did not hold element-wise:
�
Jdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/Const_2Const**
value!B Bx (dnn/head/ToFloat:0) = *
dtype0*
_output_shapes
: 
�
Udnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/SwitchSwitch?dnn/head/assert_range/assert_non_negative/assert_less_equal/All?dnn/head/assert_range/assert_non_negative/assert_less_equal/All*
T0
*
_output_shapes
: : 
�
Wdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_tIdentityWdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
�
Wdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_fIdentityUdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Switch*
T0
*
_output_shapes
: 
�
Vdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_idIdentity?dnn/head/assert_range/assert_non_negative/assert_less_equal/All*
T0
*
_output_shapes
: 
�
Sdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOpNoOpX^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_t
�
adnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/control_dependencyIdentityWdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_tT^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOp*
T0
*j
_class`
^\loc:@dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_t*
_output_shapes
: 
�
\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0ConstX^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_f*$
valueB BLabel IDs must >= 0*
dtype0*
_output_shapes
: 
�
\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1ConstX^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_f*<
value3B1 B+Condition x >= 0 did not hold element-wise:*
dtype0*
_output_shapes
: 
�
\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2ConstX^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_f**
value!B Bx (dnn/head/ToFloat:0) = *
dtype0*
_output_shapes
: 
�
Udnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/AssertAssert\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/Switch\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/Switch_1*
T
2*
	summarize
�
\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/SwitchSwitch?dnn/head/assert_range/assert_non_negative/assert_less_equal/AllVdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_id*
T0
*R
_classH
FDloc:@dnn/head/assert_range/assert_non_negative/assert_less_equal/All*
_output_shapes
: : 
�
^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/Switch_1Switchdnn/head/ToFloatVdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_id*:
_output_shapes(
&:���������:���������*
T0*#
_class
loc:@dnn/head/ToFloat
�
cdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/control_dependency_1IdentityWdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_fV^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert*
_output_shapes
: *
T0
*j
_class`
^\loc:@dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_f
�
Tdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/MergeMergecdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/control_dependency_1adnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/control_dependency*
T0
*
N*
_output_shapes
: : 
�
dnn/head/assert_range/IdentityIdentitydnn/head/ToFloat;^dnn/head/assert_range/assert_less/Assert/AssertGuard/MergeU^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Merge*
T0*'
_output_shapes
:���������
t
!dnn/head/logistic_loss/zeros_like	ZerosLikednn/logits/BiasAdd*'
_output_shapes
:���������*
T0
�
#dnn/head/logistic_loss/GreaterEqualGreaterEqualdnn/logits/BiasAdd!dnn/head/logistic_loss/zeros_like*'
_output_shapes
:���������*
T0
�
dnn/head/logistic_loss/SelectSelect#dnn/head/logistic_loss/GreaterEqualdnn/logits/BiasAdd!dnn/head/logistic_loss/zeros_like*
T0*'
_output_shapes
:���������
g
dnn/head/logistic_loss/NegNegdnn/logits/BiasAdd*
T0*'
_output_shapes
:���������
�
dnn/head/logistic_loss/Select_1Select#dnn/head/logistic_loss/GreaterEqualdnn/head/logistic_loss/Negdnn/logits/BiasAdd*
T0*'
_output_shapes
:���������
�
dnn/head/logistic_loss/mulMuldnn/logits/BiasAdddnn/head/assert_range/Identity*'
_output_shapes
:���������*
T0
�
dnn/head/logistic_loss/subSubdnn/head/logistic_loss/Selectdnn/head/logistic_loss/mul*'
_output_shapes
:���������*
T0
t
dnn/head/logistic_loss/ExpExpdnn/head/logistic_loss/Select_1*
T0*'
_output_shapes
:���������
s
dnn/head/logistic_loss/Log1pLog1pdnn/head/logistic_loss/Exp*
T0*'
_output_shapes
:���������
�
dnn/head/logistic_lossAdddnn/head/logistic_loss/subdnn/head/logistic_loss/Log1p*
T0*'
_output_shapes
:���������
x
3dnn/head/weighted_loss/assert_broadcastable/weightsConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
|
9dnn/head/weighted_loss/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
z
8dnn/head/weighted_loss/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
�
8dnn/head/weighted_loss/assert_broadcastable/values/shapeShapednn/head/logistic_loss*
out_type0*
_output_shapes
:*
T0
y
7dnn/head/weighted_loss/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
O
Gdnn/head/weighted_loss/assert_broadcastable/static_scalar_check_successNoOp
�
"dnn/head/weighted_loss/ToFloat_1/xConstH^dnn/head/weighted_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
dnn/head/weighted_loss/MulMuldnn/head/logistic_loss"dnn/head/weighted_loss/ToFloat_1/x*
T0*'
_output_shapes
:���������
�
dnn/head/weighted_loss/ConstConstH^dnn/head/weighted_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
:*
valueB"       *
dtype0
�
dnn/head/weighted_loss/SumSumdnn/head/weighted_loss/Muldnn/head/weighted_loss/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
z
5dnn/head/metrics/label/mean/broadcast_weights/weightsConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Pdnn/head/metrics/label/mean/broadcast_weights/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Odnn/head/metrics/label/mean/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
�
Odnn/head/metrics/label/mean/broadcast_weights/assert_broadcastable/values/shapeShapednn/head/assert_range/Identity*
T0*
out_type0*
_output_shapes
:
�
Ndnn/head/metrics/label/mean/broadcast_weights/assert_broadcastable/values/rankConst*
_output_shapes
: *
value	B :*
dtype0
f
^dnn/head/metrics/label/mean/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
�
=dnn/head/metrics/label/mean/broadcast_weights/ones_like/ShapeShapednn/head/assert_range/Identity_^dnn/head/metrics/label/mean/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
�
=dnn/head/metrics/label/mean/broadcast_weights/ones_like/ConstConst_^dnn/head/metrics/label/mean/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
7dnn/head/metrics/label/mean/broadcast_weights/ones_likeFill=dnn/head/metrics/label/mean/broadcast_weights/ones_like/Shape=dnn/head/metrics/label/mean/broadcast_weights/ones_like/Const*

index_type0*'
_output_shapes
:���������*
T0
�
-dnn/head/metrics/label/mean/broadcast_weightsMul5dnn/head/metrics/label/mean/broadcast_weights/weights7dnn/head/metrics/label/mean/broadcast_weights/ones_like*
T0*'
_output_shapes
:���������
�
3dnn/head/metrics/label/mean/total/Initializer/zerosConst*4
_class*
(&loc:@dnn/head/metrics/label/mean/total*
valueB
 *    *
dtype0*
_output_shapes
: 
�
!dnn/head/metrics/label/mean/total
VariableV2*
dtype0*
_output_shapes
: *
shared_name *4
_class*
(&loc:@dnn/head/metrics/label/mean/total*
	container *
shape: 
�
(dnn/head/metrics/label/mean/total/AssignAssign!dnn/head/metrics/label/mean/total3dnn/head/metrics/label/mean/total/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*4
_class*
(&loc:@dnn/head/metrics/label/mean/total
�
&dnn/head/metrics/label/mean/total/readIdentity!dnn/head/metrics/label/mean/total*4
_class*
(&loc:@dnn/head/metrics/label/mean/total*
_output_shapes
: *
T0
�
3dnn/head/metrics/label/mean/count/Initializer/zerosConst*
_output_shapes
: *4
_class*
(&loc:@dnn/head/metrics/label/mean/count*
valueB
 *    *
dtype0
�
!dnn/head/metrics/label/mean/count
VariableV2*
shared_name *4
_class*
(&loc:@dnn/head/metrics/label/mean/count*
	container *
shape: *
dtype0*
_output_shapes
: 
�
(dnn/head/metrics/label/mean/count/AssignAssign!dnn/head/metrics/label/mean/count3dnn/head/metrics/label/mean/count/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@dnn/head/metrics/label/mean/count*
validate_shape(*
_output_shapes
: 
�
&dnn/head/metrics/label/mean/count/readIdentity!dnn/head/metrics/label/mean/count*
T0*4
_class*
(&loc:@dnn/head/metrics/label/mean/count*
_output_shapes
: 
�
Rdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/shapeShape-dnn/head/metrics/label/mean/broadcast_weights*
_output_shapes
:*
T0*
out_type0
�
Qdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/rankConst*
value	B :*
dtype0*
_output_shapes
: 
�
Qdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/shapeShapednn/head/assert_range/Identity*
T0*
out_type0*
_output_shapes
:
�
Pdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
�
Pdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalar/xConst*
value	B : *
dtype0*
_output_shapes
: 
�
Ndnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalarEqualPdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalar/xQdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/rank*
_output_shapes
: *
T0
�
Zdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/SwitchSwitchNdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalarNdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalar*
_output_shapes
: : *
T0

�
\dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_tIdentity\dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch:1*
_output_shapes
: *
T0

�
\dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_fIdentityZdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch*
_output_shapes
: *
T0

�
[dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_idIdentityNdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: 
�
\dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1SwitchNdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalar[dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0
*a
_classW
USloc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalar*
_output_shapes
: : 
�
zdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqual�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1*
T0*
_output_shapes
: 
�
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchSwitchPdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/rank[dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*c
_classY
WUloc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/rank*
_output_shapes
: : *
T0
�
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1SwitchQdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/rank[dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
_output_shapes
: : *
T0*d
_classZ
XVloc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/rank
�
tdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/SwitchSwitchzdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankzdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: : 
�
vdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_tIdentityvdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1*
T0
*
_output_shapes
: 
�
vdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_fIdentitytdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch*
T0
*
_output_shapes
: 
�
udnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_idIdentityzdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: 
�
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConstw^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
���������*
dtype0*
_output_shapes
: 
�
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDims�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim*

Tdim0*
T0*
_output_shapes

:
�
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchSwitchQdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/shape[dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id* 
_output_shapes
::*
T0*d
_classZ
XVloc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/shape
�
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1Switch�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switchudnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*d
_classZ
XVloc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/shape* 
_output_shapes
::
�
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeConstw^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB"      *
dtype0*
_output_shapes
:
�
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConstw^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
�
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFill�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const*

index_type0*
_output_shapes

:*
T0
�
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConstw^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
�
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis*
T0*
N*
_output_shapes

:*

Tidx0
�
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConstw^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
���������*
dtype0*
_output_shapes
: 
�
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDims�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim*
_output_shapes

:*

Tdim0*
T0
�
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchSwitchRdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/shape[dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id* 
_output_shapes
::*
T0*e
_class[
YWloc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/shape
�
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1Switch�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switchudnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*e
_class[
YWloc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/shape* 
_output_shapes
::
�
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperation�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat*
validate_indices(*
T0*<
_output_shapes*
(:���������:���������:*
set_operationa-b
�
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSize�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1*
_output_shapes
: *
T0*
out_type0
�
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConstw^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B : *
dtype0*
_output_shapes
: 
�
~dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqual�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims*
T0*
_output_shapes
: 
�
vdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Switchzdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankudnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
_output_shapes
: : *
T0
*�
_class�
�loc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank
�
sdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeMergevdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1~dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims*
T0
*
N*
_output_shapes
: : 
�
Ydnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/MergeMergesdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1:1*
T0
*
N*
_output_shapes
: : 
�
Jdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/ConstConst*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
�
Ldnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/Const_1Const*
_output_shapes
: *
valueB Bweights.shape=*
dtype0
�
Ldnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/Const_2Const*@
value7B5 B/dnn/head/metrics/label/mean/broadcast_weights:0*
dtype0*
_output_shapes
: 
�
Ldnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/Const_3Const*
_output_shapes
: *
valueB Bvalues.shape=*
dtype0
�
Ldnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/Const_4Const*1
value(B& B dnn/head/assert_range/Identity:0*
dtype0*
_output_shapes
: 
�
Ldnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/Const_5Const*
dtype0*
_output_shapes
: *
valueB B
is_scalar=
�
Wdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/SwitchSwitchYdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/MergeYdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: : *
T0

�
Ydnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_tIdentityYdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
�
Ydnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_fIdentityWdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Switch*
T0
*
_output_shapes
: 
�
Xdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_idIdentityYdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: 
�
Udnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/NoOpNoOpZ^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t
�
cdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependencyIdentityYdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_tV^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/NoOp*
T0
*l
_classb
`^loc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t*
_output_shapes
: 
�
^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0ConstZ^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
�
^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1ConstZ^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
�
^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2ConstZ^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*@
value7B5 B/dnn/head/metrics/label/mean/broadcast_weights:0*
dtype0*
_output_shapes
: 
�
^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4ConstZ^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
�
^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5ConstZ^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*1
value(B& B dnn/head/assert_range/Identity:0*
dtype0*
_output_shapes
: 
�
^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7ConstZ^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
dtype0*
_output_shapes
: *
valueB B
is_scalar=
�
Wdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/AssertAssert^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2`dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5`dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7`dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3*
T
2	
*
	summarize
�
^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/SwitchSwitchYdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/MergeXdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*l
_classb
`^loc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: : *
T0

�
`dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1SwitchRdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/shapeXdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id* 
_output_shapes
::*
T0*e
_class[
YWloc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/shape
�
`dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2SwitchQdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/shapeXdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0*d
_classZ
XVloc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/shape* 
_output_shapes
::
�
`dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3SwitchNdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalarXdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0
*a
_classW
USloc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalar*
_output_shapes
: : 
�
ednn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency_1IdentityYdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_fX^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert*
T0
*l
_classb
`^loc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: 
�
Vdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/MergeMergeednn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency_1cdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency*
T0
*
N*
_output_shapes
: : 
�
?dnn/head/metrics/label/mean/broadcast_weights_1/ones_like/ShapeShapednn/head/assert_range/IdentityW^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Merge*
T0*
out_type0*
_output_shapes
:
�
?dnn/head/metrics/label/mean/broadcast_weights_1/ones_like/ConstConstW^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Merge*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
9dnn/head/metrics/label/mean/broadcast_weights_1/ones_likeFill?dnn/head/metrics/label/mean/broadcast_weights_1/ones_like/Shape?dnn/head/metrics/label/mean/broadcast_weights_1/ones_like/Const*'
_output_shapes
:���������*
T0*

index_type0
�
/dnn/head/metrics/label/mean/broadcast_weights_1Mul-dnn/head/metrics/label/mean/broadcast_weights9dnn/head/metrics/label/mean/broadcast_weights_1/ones_like*
T0*'
_output_shapes
:���������
�
dnn/head/metrics/label/mean/MulMuldnn/head/assert_range/Identity/dnn/head/metrics/label/mean/broadcast_weights_1*
T0*'
_output_shapes
:���������
r
!dnn/head/metrics/label/mean/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
dnn/head/metrics/label/mean/SumSum/dnn/head/metrics/label/mean/broadcast_weights_1!dnn/head/metrics/label/mean/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
t
#dnn/head/metrics/label/mean/Const_1Const*
_output_shapes
:*
valueB"       *
dtype0
�
!dnn/head/metrics/label/mean/Sum_1Sumdnn/head/metrics/label/mean/Mul#dnn/head/metrics/label/mean/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
%dnn/head/metrics/label/mean/AssignAdd	AssignAdd!dnn/head/metrics/label/mean/total!dnn/head/metrics/label/mean/Sum_1*
T0*4
_class*
(&loc:@dnn/head/metrics/label/mean/total*
_output_shapes
: *
use_locking( 
�
'dnn/head/metrics/label/mean/AssignAdd_1	AssignAdd!dnn/head/metrics/label/mean/countdnn/head/metrics/label/mean/Sum ^dnn/head/metrics/label/mean/Mul*
use_locking( *
T0*4
_class*
(&loc:@dnn/head/metrics/label/mean/count*
_output_shapes
: 
�
#dnn/head/metrics/label/mean/truedivRealDiv&dnn/head/metrics/label/mean/total/read&dnn/head/metrics/label/mean/count/read*
T0*
_output_shapes
: 
k
&dnn/head/metrics/label/mean/zeros_likeConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
#dnn/head/metrics/label/mean/GreaterGreater&dnn/head/metrics/label/mean/count/read&dnn/head/metrics/label/mean/zeros_like*
_output_shapes
: *
T0
�
!dnn/head/metrics/label/mean/valueSelect#dnn/head/metrics/label/mean/Greater#dnn/head/metrics/label/mean/truediv&dnn/head/metrics/label/mean/zeros_like*
T0*
_output_shapes
: 
�
%dnn/head/metrics/label/mean/truediv_1RealDiv%dnn/head/metrics/label/mean/AssignAdd'dnn/head/metrics/label/mean/AssignAdd_1*
T0*
_output_shapes
: 
m
(dnn/head/metrics/label/mean/zeros_like_1Const*
valueB
 *    *
dtype0*
_output_shapes
: 
�
%dnn/head/metrics/label/mean/Greater_1Greater'dnn/head/metrics/label/mean/AssignAdd_1(dnn/head/metrics/label/mean/zeros_like_1*
T0*
_output_shapes
: 
�
%dnn/head/metrics/label/mean/update_opSelect%dnn/head/metrics/label/mean/Greater_1%dnn/head/metrics/label/mean/truediv_1(dnn/head/metrics/label/mean/zeros_like_1*
T0*
_output_shapes
: 
�
5dnn/head/metrics/average_loss/total/Initializer/zerosConst*6
_class,
*(loc:@dnn/head/metrics/average_loss/total*
valueB
 *    *
dtype0*
_output_shapes
: 
�
#dnn/head/metrics/average_loss/total
VariableV2*
dtype0*
_output_shapes
: *
shared_name *6
_class,
*(loc:@dnn/head/metrics/average_loss/total*
	container *
shape: 
�
*dnn/head/metrics/average_loss/total/AssignAssign#dnn/head/metrics/average_loss/total5dnn/head/metrics/average_loss/total/Initializer/zeros*
T0*6
_class,
*(loc:@dnn/head/metrics/average_loss/total*
validate_shape(*
_output_shapes
: *
use_locking(
�
(dnn/head/metrics/average_loss/total/readIdentity#dnn/head/metrics/average_loss/total*
T0*6
_class,
*(loc:@dnn/head/metrics/average_loss/total*
_output_shapes
: 
�
5dnn/head/metrics/average_loss/count/Initializer/zerosConst*
_output_shapes
: *6
_class,
*(loc:@dnn/head/metrics/average_loss/count*
valueB
 *    *
dtype0
�
#dnn/head/metrics/average_loss/count
VariableV2*6
_class,
*(loc:@dnn/head/metrics/average_loss/count*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
�
*dnn/head/metrics/average_loss/count/AssignAssign#dnn/head/metrics/average_loss/count5dnn/head/metrics/average_loss/count/Initializer/zeros*6
_class,
*(loc:@dnn/head/metrics/average_loss/count*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
(dnn/head/metrics/average_loss/count/readIdentity#dnn/head/metrics/average_loss/count*
T0*6
_class,
*(loc:@dnn/head/metrics/average_loss/count*
_output_shapes
: 
h
#dnn/head/metrics/average_loss/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Rdnn/head/metrics/average_loss/broadcast_weights/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Qdnn/head/metrics/average_loss/broadcast_weights/assert_broadcastable/weights/rankConst*
dtype0*
_output_shapes
: *
value	B : 
�
Qdnn/head/metrics/average_loss/broadcast_weights/assert_broadcastable/values/shapeShapednn/head/logistic_loss*
out_type0*
_output_shapes
:*
T0
�
Pdnn/head/metrics/average_loss/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
h
`dnn/head/metrics/average_loss/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
�
?dnn/head/metrics/average_loss/broadcast_weights/ones_like/ShapeShapednn/head/logistic_lossa^dnn/head/metrics/average_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
�
?dnn/head/metrics/average_loss/broadcast_weights/ones_like/ConstConsta^dnn/head/metrics/average_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
9dnn/head/metrics/average_loss/broadcast_weights/ones_likeFill?dnn/head/metrics/average_loss/broadcast_weights/ones_like/Shape?dnn/head/metrics/average_loss/broadcast_weights/ones_like/Const*
T0*

index_type0*'
_output_shapes
:���������
�
/dnn/head/metrics/average_loss/broadcast_weightsMul#dnn/head/metrics/average_loss/Const9dnn/head/metrics/average_loss/broadcast_weights/ones_like*'
_output_shapes
:���������*
T0
�
!dnn/head/metrics/average_loss/MulMuldnn/head/logistic_loss/dnn/head/metrics/average_loss/broadcast_weights*
T0*'
_output_shapes
:���������
v
%dnn/head/metrics/average_loss/Const_1Const*
valueB"       *
dtype0*
_output_shapes
:
�
!dnn/head/metrics/average_loss/SumSum/dnn/head/metrics/average_loss/broadcast_weights%dnn/head/metrics/average_loss/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
v
%dnn/head/metrics/average_loss/Const_2Const*
valueB"       *
dtype0*
_output_shapes
:
�
#dnn/head/metrics/average_loss/Sum_1Sum!dnn/head/metrics/average_loss/Mul%dnn/head/metrics/average_loss/Const_2*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
'dnn/head/metrics/average_loss/AssignAdd	AssignAdd#dnn/head/metrics/average_loss/total#dnn/head/metrics/average_loss/Sum_1*
_output_shapes
: *
use_locking( *
T0*6
_class,
*(loc:@dnn/head/metrics/average_loss/total
�
)dnn/head/metrics/average_loss/AssignAdd_1	AssignAdd#dnn/head/metrics/average_loss/count!dnn/head/metrics/average_loss/Sum"^dnn/head/metrics/average_loss/Mul*
use_locking( *
T0*6
_class,
*(loc:@dnn/head/metrics/average_loss/count*
_output_shapes
: 
�
%dnn/head/metrics/average_loss/truedivRealDiv(dnn/head/metrics/average_loss/total/read(dnn/head/metrics/average_loss/count/read*
_output_shapes
: *
T0
m
(dnn/head/metrics/average_loss/zeros_likeConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
%dnn/head/metrics/average_loss/GreaterGreater(dnn/head/metrics/average_loss/count/read(dnn/head/metrics/average_loss/zeros_like*
T0*
_output_shapes
: 
�
#dnn/head/metrics/average_loss/valueSelect%dnn/head/metrics/average_loss/Greater%dnn/head/metrics/average_loss/truediv(dnn/head/metrics/average_loss/zeros_like*
T0*
_output_shapes
: 
�
'dnn/head/metrics/average_loss/truediv_1RealDiv'dnn/head/metrics/average_loss/AssignAdd)dnn/head/metrics/average_loss/AssignAdd_1*
_output_shapes
: *
T0
o
*dnn/head/metrics/average_loss/zeros_like_1Const*
valueB
 *    *
dtype0*
_output_shapes
: 
�
'dnn/head/metrics/average_loss/Greater_1Greater)dnn/head/metrics/average_loss/AssignAdd_1*dnn/head/metrics/average_loss/zeros_like_1*
T0*
_output_shapes
: 
�
'dnn/head/metrics/average_loss/update_opSelect'dnn/head/metrics/average_loss/Greater_1'dnn/head/metrics/average_loss/truediv_1*dnn/head/metrics/average_loss/zeros_like_1*
T0*
_output_shapes
: 
[
dnn/head/metrics/ConstConst*
_output_shapes
: *
valueB
 *  �?*
dtype0

dnn/head/metrics/CastCastdnn/head/predictions/ExpandDims*

SrcT0	*'
_output_shapes
:���������*

DstT0
�
dnn/head/metrics/EqualEqualdnn/head/metrics/Castdnn/head/assert_range/Identity*'
_output_shapes
:���������*
T0
y
dnn/head/metrics/ToFloatCastdnn/head/metrics/Equal*'
_output_shapes
:���������*

DstT0*

SrcT0

�
1dnn/head/metrics/accuracy/total/Initializer/zerosConst*2
_class(
&$loc:@dnn/head/metrics/accuracy/total*
valueB
 *    *
dtype0*
_output_shapes
: 
�
dnn/head/metrics/accuracy/total
VariableV2*2
_class(
&$loc:@dnn/head/metrics/accuracy/total*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
�
&dnn/head/metrics/accuracy/total/AssignAssigndnn/head/metrics/accuracy/total1dnn/head/metrics/accuracy/total/Initializer/zeros*
use_locking(*
T0*2
_class(
&$loc:@dnn/head/metrics/accuracy/total*
validate_shape(*
_output_shapes
: 
�
$dnn/head/metrics/accuracy/total/readIdentitydnn/head/metrics/accuracy/total*
_output_shapes
: *
T0*2
_class(
&$loc:@dnn/head/metrics/accuracy/total
�
1dnn/head/metrics/accuracy/count/Initializer/zerosConst*2
_class(
&$loc:@dnn/head/metrics/accuracy/count*
valueB
 *    *
dtype0*
_output_shapes
: 
�
dnn/head/metrics/accuracy/count
VariableV2*
shared_name *2
_class(
&$loc:@dnn/head/metrics/accuracy/count*
	container *
shape: *
dtype0*
_output_shapes
: 
�
&dnn/head/metrics/accuracy/count/AssignAssigndnn/head/metrics/accuracy/count1dnn/head/metrics/accuracy/count/Initializer/zeros*
use_locking(*
T0*2
_class(
&$loc:@dnn/head/metrics/accuracy/count*
validate_shape(*
_output_shapes
: 
�
$dnn/head/metrics/accuracy/count/readIdentitydnn/head/metrics/accuracy/count*
_output_shapes
: *
T0*2
_class(
&$loc:@dnn/head/metrics/accuracy/count
�
Ndnn/head/metrics/accuracy/broadcast_weights/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Mdnn/head/metrics/accuracy/broadcast_weights/assert_broadcastable/weights/rankConst*
_output_shapes
: *
value	B : *
dtype0
�
Mdnn/head/metrics/accuracy/broadcast_weights/assert_broadcastable/values/shapeShapednn/head/metrics/ToFloat*
T0*
out_type0*
_output_shapes
:
�
Ldnn/head/metrics/accuracy/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
d
\dnn/head/metrics/accuracy/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
�
;dnn/head/metrics/accuracy/broadcast_weights/ones_like/ShapeShapednn/head/metrics/ToFloat]^dnn/head/metrics/accuracy/broadcast_weights/assert_broadcastable/static_scalar_check_success*
out_type0*
_output_shapes
:*
T0
�
;dnn/head/metrics/accuracy/broadcast_weights/ones_like/ConstConst]^dnn/head/metrics/accuracy/broadcast_weights/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
valueB
 *  �?*
dtype0
�
5dnn/head/metrics/accuracy/broadcast_weights/ones_likeFill;dnn/head/metrics/accuracy/broadcast_weights/ones_like/Shape;dnn/head/metrics/accuracy/broadcast_weights/ones_like/Const*
T0*

index_type0*'
_output_shapes
:���������
�
+dnn/head/metrics/accuracy/broadcast_weightsMuldnn/head/metrics/Const5dnn/head/metrics/accuracy/broadcast_weights/ones_like*
T0*'
_output_shapes
:���������
�
dnn/head/metrics/accuracy/MulMuldnn/head/metrics/ToFloat+dnn/head/metrics/accuracy/broadcast_weights*'
_output_shapes
:���������*
T0
p
dnn/head/metrics/accuracy/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
dnn/head/metrics/accuracy/SumSum+dnn/head/metrics/accuracy/broadcast_weightsdnn/head/metrics/accuracy/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
r
!dnn/head/metrics/accuracy/Const_1Const*
valueB"       *
dtype0*
_output_shapes
:
�
dnn/head/metrics/accuracy/Sum_1Sumdnn/head/metrics/accuracy/Mul!dnn/head/metrics/accuracy/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
#dnn/head/metrics/accuracy/AssignAdd	AssignAdddnn/head/metrics/accuracy/totaldnn/head/metrics/accuracy/Sum_1*
use_locking( *
T0*2
_class(
&$loc:@dnn/head/metrics/accuracy/total*
_output_shapes
: 
�
%dnn/head/metrics/accuracy/AssignAdd_1	AssignAdddnn/head/metrics/accuracy/countdnn/head/metrics/accuracy/Sum^dnn/head/metrics/accuracy/Mul*
use_locking( *
T0*2
_class(
&$loc:@dnn/head/metrics/accuracy/count*
_output_shapes
: 
�
!dnn/head/metrics/accuracy/truedivRealDiv$dnn/head/metrics/accuracy/total/read$dnn/head/metrics/accuracy/count/read*
T0*
_output_shapes
: 
i
$dnn/head/metrics/accuracy/zeros_likeConst*
_output_shapes
: *
valueB
 *    *
dtype0
�
!dnn/head/metrics/accuracy/GreaterGreater$dnn/head/metrics/accuracy/count/read$dnn/head/metrics/accuracy/zeros_like*
T0*
_output_shapes
: 
�
dnn/head/metrics/accuracy/valueSelect!dnn/head/metrics/accuracy/Greater!dnn/head/metrics/accuracy/truediv$dnn/head/metrics/accuracy/zeros_like*
T0*
_output_shapes
: 
�
#dnn/head/metrics/accuracy/truediv_1RealDiv#dnn/head/metrics/accuracy/AssignAdd%dnn/head/metrics/accuracy/AssignAdd_1*
T0*
_output_shapes
: 
k
&dnn/head/metrics/accuracy/zeros_like_1Const*
valueB
 *    *
dtype0*
_output_shapes
: 
�
#dnn/head/metrics/accuracy/Greater_1Greater%dnn/head/metrics/accuracy/AssignAdd_1&dnn/head/metrics/accuracy/zeros_like_1*
_output_shapes
: *
T0
�
#dnn/head/metrics/accuracy/update_opSelect#dnn/head/metrics/accuracy/Greater_1#dnn/head/metrics/accuracy/truediv_1&dnn/head/metrics/accuracy/zeros_like_1*
_output_shapes
: *
T0
�
dnn/head/metrics/precision/CastCastdnn/head/predictions/ExpandDims*'
_output_shapes
:���������*

DstT0
*

SrcT0	
�
!dnn/head/metrics/precision/Cast_1Castdnn/head/assert_range/Identity*

SrcT0*'
_output_shapes
:���������*

DstT0

e
 dnn/head/metrics/precision/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
s
1dnn/head/metrics/precision/true_positives/Equal/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 
�
/dnn/head/metrics/precision/true_positives/EqualEqual!dnn/head/metrics/precision/Cast_11dnn/head/metrics/precision/true_positives/Equal/y*'
_output_shapes
:���������*
T0

u
3dnn/head/metrics/precision/true_positives/Equal_1/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 
�
1dnn/head/metrics/precision/true_positives/Equal_1Equaldnn/head/metrics/precision/Cast3dnn/head/metrics/precision/true_positives/Equal_1/y*'
_output_shapes
:���������*
T0

�
4dnn/head/metrics/precision/true_positives/LogicalAnd
LogicalAnd/dnn/head/metrics/precision/true_positives/Equal1dnn/head/metrics/precision/true_positives/Equal_1*'
_output_shapes
:���������
`
Xdnn/head/metrics/precision/true_positives/assert_type/statically_determined_correct_typeNoOp
�
Adnn/head/metrics/precision/true_positives/count/Initializer/zerosConst*
dtype0*
_output_shapes
: *B
_class8
64loc:@dnn/head/metrics/precision/true_positives/count*
valueB
 *    
�
/dnn/head/metrics/precision/true_positives/count
VariableV2*
shared_name *B
_class8
64loc:@dnn/head/metrics/precision/true_positives/count*
	container *
shape: *
dtype0*
_output_shapes
: 
�
6dnn/head/metrics/precision/true_positives/count/AssignAssign/dnn/head/metrics/precision/true_positives/countAdnn/head/metrics/precision/true_positives/count/Initializer/zeros*B
_class8
64loc:@dnn/head/metrics/precision/true_positives/count*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
4dnn/head/metrics/precision/true_positives/count/readIdentity/dnn/head/metrics/precision/true_positives/count*
_output_shapes
: *
T0*B
_class8
64loc:@dnn/head/metrics/precision/true_positives/count
�
1dnn/head/metrics/precision/true_positives/ToFloatCast4dnn/head/metrics/precision/true_positives/LogicalAnd*'
_output_shapes
:���������*

DstT0*

SrcT0

p
.dnn/head/metrics/precision/true_positives/RankConst*
value	B :*
dtype0*
_output_shapes
: 

=dnn/head/metrics/precision/true_positives/assert_rank_in/rankConst*
value	B : *
dtype0*
_output_shapes
: 
�
>dnn/head/metrics/precision/true_positives/assert_rank_in/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
o
gdnn/head/metrics/precision/true_positives/assert_rank_in/assert_type/statically_determined_correct_typeNoOp
q
idnn/head/metrics/precision/true_positives/assert_rank_in/assert_type_1/statically_determined_correct_typeNoOp
`
Xdnn/head/metrics/precision/true_positives/assert_rank_in/static_checks_determined_all_okNoOp
�
-dnn/head/metrics/precision/true_positives/MulMul1dnn/head/metrics/precision/true_positives/ToFloat dnn/head/metrics/precision/ConstY^dnn/head/metrics/precision/true_positives/assert_rank_in/static_checks_determined_all_ok*'
_output_shapes
:���������*
T0
�
2dnn/head/metrics/precision/true_positives/IdentityIdentity4dnn/head/metrics/precision/true_positives/count/read*
_output_shapes
: *
T0
�
/dnn/head/metrics/precision/true_positives/ConstConst*
dtype0*
_output_shapes
:*
valueB"       
�
-dnn/head/metrics/precision/true_positives/SumSum-dnn/head/metrics/precision/true_positives/Mul/dnn/head/metrics/precision/true_positives/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
3dnn/head/metrics/precision/true_positives/AssignAdd	AssignAdd/dnn/head/metrics/precision/true_positives/count-dnn/head/metrics/precision/true_positives/Sum*
_output_shapes
: *
use_locking( *
T0*B
_class8
64loc:@dnn/head/metrics/precision/true_positives/count
t
2dnn/head/metrics/precision/false_positives/Equal/yConst*
value	B
 Z *
dtype0
*
_output_shapes
: 
�
0dnn/head/metrics/precision/false_positives/EqualEqual!dnn/head/metrics/precision/Cast_12dnn/head/metrics/precision/false_positives/Equal/y*
T0
*'
_output_shapes
:���������
v
4dnn/head/metrics/precision/false_positives/Equal_1/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 
�
2dnn/head/metrics/precision/false_positives/Equal_1Equaldnn/head/metrics/precision/Cast4dnn/head/metrics/precision/false_positives/Equal_1/y*
T0
*'
_output_shapes
:���������
�
5dnn/head/metrics/precision/false_positives/LogicalAnd
LogicalAnd0dnn/head/metrics/precision/false_positives/Equal2dnn/head/metrics/precision/false_positives/Equal_1*'
_output_shapes
:���������
a
Ydnn/head/metrics/precision/false_positives/assert_type/statically_determined_correct_typeNoOp
�
Bdnn/head/metrics/precision/false_positives/count/Initializer/zerosConst*C
_class9
75loc:@dnn/head/metrics/precision/false_positives/count*
valueB
 *    *
dtype0*
_output_shapes
: 
�
0dnn/head/metrics/precision/false_positives/count
VariableV2*
dtype0*
_output_shapes
: *
shared_name *C
_class9
75loc:@dnn/head/metrics/precision/false_positives/count*
	container *
shape: 
�
7dnn/head/metrics/precision/false_positives/count/AssignAssign0dnn/head/metrics/precision/false_positives/countBdnn/head/metrics/precision/false_positives/count/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*C
_class9
75loc:@dnn/head/metrics/precision/false_positives/count
�
5dnn/head/metrics/precision/false_positives/count/readIdentity0dnn/head/metrics/precision/false_positives/count*
_output_shapes
: *
T0*C
_class9
75loc:@dnn/head/metrics/precision/false_positives/count
�
2dnn/head/metrics/precision/false_positives/ToFloatCast5dnn/head/metrics/precision/false_positives/LogicalAnd*

SrcT0
*'
_output_shapes
:���������*

DstT0
q
/dnn/head/metrics/precision/false_positives/RankConst*
value	B :*
dtype0*
_output_shapes
: 
�
>dnn/head/metrics/precision/false_positives/assert_rank_in/rankConst*
value	B : *
dtype0*
_output_shapes
: 
�
?dnn/head/metrics/precision/false_positives/assert_rank_in/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
p
hdnn/head/metrics/precision/false_positives/assert_rank_in/assert_type/statically_determined_correct_typeNoOp
r
jdnn/head/metrics/precision/false_positives/assert_rank_in/assert_type_1/statically_determined_correct_typeNoOp
a
Ydnn/head/metrics/precision/false_positives/assert_rank_in/static_checks_determined_all_okNoOp
�
.dnn/head/metrics/precision/false_positives/MulMul2dnn/head/metrics/precision/false_positives/ToFloat dnn/head/metrics/precision/ConstZ^dnn/head/metrics/precision/false_positives/assert_rank_in/static_checks_determined_all_ok*'
_output_shapes
:���������*
T0
�
3dnn/head/metrics/precision/false_positives/IdentityIdentity5dnn/head/metrics/precision/false_positives/count/read*
T0*
_output_shapes
: 
�
0dnn/head/metrics/precision/false_positives/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
.dnn/head/metrics/precision/false_positives/SumSum.dnn/head/metrics/precision/false_positives/Mul0dnn/head/metrics/precision/false_positives/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
4dnn/head/metrics/precision/false_positives/AssignAdd	AssignAdd0dnn/head/metrics/precision/false_positives/count.dnn/head/metrics/precision/false_positives/Sum*C
_class9
75loc:@dnn/head/metrics/precision/false_positives/count*
_output_shapes
: *
use_locking( *
T0
�
dnn/head/metrics/precision/addAdd2dnn/head/metrics/precision/true_positives/Identity3dnn/head/metrics/precision/false_positives/Identity*
T0*
_output_shapes
: 
i
$dnn/head/metrics/precision/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
"dnn/head/metrics/precision/GreaterGreaterdnn/head/metrics/precision/add$dnn/head/metrics/precision/Greater/y*
T0*
_output_shapes
: 
�
 dnn/head/metrics/precision/add_1Add2dnn/head/metrics/precision/true_positives/Identity3dnn/head/metrics/precision/false_positives/Identity*
_output_shapes
: *
T0
�
dnn/head/metrics/precision/divRealDiv2dnn/head/metrics/precision/true_positives/Identity dnn/head/metrics/precision/add_1*
_output_shapes
: *
T0
g
"dnn/head/metrics/precision/value/eConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
 dnn/head/metrics/precision/valueSelect"dnn/head/metrics/precision/Greaterdnn/head/metrics/precision/div"dnn/head/metrics/precision/value/e*
T0*
_output_shapes
: 
�
 dnn/head/metrics/precision/add_2Add3dnn/head/metrics/precision/true_positives/AssignAdd4dnn/head/metrics/precision/false_positives/AssignAdd*
T0*
_output_shapes
: 
k
&dnn/head/metrics/precision/Greater_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
$dnn/head/metrics/precision/Greater_1Greater dnn/head/metrics/precision/add_2&dnn/head/metrics/precision/Greater_1/y*
_output_shapes
: *
T0
�
 dnn/head/metrics/precision/add_3Add3dnn/head/metrics/precision/true_positives/AssignAdd4dnn/head/metrics/precision/false_positives/AssignAdd*
T0*
_output_shapes
: 
�
 dnn/head/metrics/precision/div_1RealDiv3dnn/head/metrics/precision/true_positives/AssignAdd dnn/head/metrics/precision/add_3*
_output_shapes
: *
T0
k
&dnn/head/metrics/precision/update_op/eConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
$dnn/head/metrics/precision/update_opSelect$dnn/head/metrics/precision/Greater_1 dnn/head/metrics/precision/div_1&dnn/head/metrics/precision/update_op/e*
_output_shapes
: *
T0
�
dnn/head/metrics/recall/CastCastdnn/head/predictions/ExpandDims*'
_output_shapes
:���������*

DstT0
*

SrcT0	
�
dnn/head/metrics/recall/Cast_1Castdnn/head/assert_range/Identity*

SrcT0*'
_output_shapes
:���������*

DstT0

b
dnn/head/metrics/recall/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
p
.dnn/head/metrics/recall/true_positives/Equal/yConst*
_output_shapes
: *
value	B
 Z*
dtype0

�
,dnn/head/metrics/recall/true_positives/EqualEqualdnn/head/metrics/recall/Cast_1.dnn/head/metrics/recall/true_positives/Equal/y*'
_output_shapes
:���������*
T0

r
0dnn/head/metrics/recall/true_positives/Equal_1/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 
�
.dnn/head/metrics/recall/true_positives/Equal_1Equaldnn/head/metrics/recall/Cast0dnn/head/metrics/recall/true_positives/Equal_1/y*
T0
*'
_output_shapes
:���������
�
1dnn/head/metrics/recall/true_positives/LogicalAnd
LogicalAnd,dnn/head/metrics/recall/true_positives/Equal.dnn/head/metrics/recall/true_positives/Equal_1*'
_output_shapes
:���������
]
Udnn/head/metrics/recall/true_positives/assert_type/statically_determined_correct_typeNoOp
�
>dnn/head/metrics/recall/true_positives/count/Initializer/zerosConst*
_output_shapes
: *?
_class5
31loc:@dnn/head/metrics/recall/true_positives/count*
valueB
 *    *
dtype0
�
,dnn/head/metrics/recall/true_positives/count
VariableV2*
dtype0*
_output_shapes
: *
shared_name *?
_class5
31loc:@dnn/head/metrics/recall/true_positives/count*
	container *
shape: 
�
3dnn/head/metrics/recall/true_positives/count/AssignAssign,dnn/head/metrics/recall/true_positives/count>dnn/head/metrics/recall/true_positives/count/Initializer/zeros*
_output_shapes
: *
use_locking(*
T0*?
_class5
31loc:@dnn/head/metrics/recall/true_positives/count*
validate_shape(
�
1dnn/head/metrics/recall/true_positives/count/readIdentity,dnn/head/metrics/recall/true_positives/count*
T0*?
_class5
31loc:@dnn/head/metrics/recall/true_positives/count*
_output_shapes
: 
�
.dnn/head/metrics/recall/true_positives/ToFloatCast1dnn/head/metrics/recall/true_positives/LogicalAnd*

SrcT0
*'
_output_shapes
:���������*

DstT0
m
+dnn/head/metrics/recall/true_positives/RankConst*
_output_shapes
: *
value	B :*
dtype0
|
:dnn/head/metrics/recall/true_positives/assert_rank_in/rankConst*
value	B : *
dtype0*
_output_shapes
: 
~
;dnn/head/metrics/recall/true_positives/assert_rank_in/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
l
ddnn/head/metrics/recall/true_positives/assert_rank_in/assert_type/statically_determined_correct_typeNoOp
n
fdnn/head/metrics/recall/true_positives/assert_rank_in/assert_type_1/statically_determined_correct_typeNoOp
]
Udnn/head/metrics/recall/true_positives/assert_rank_in/static_checks_determined_all_okNoOp
�
*dnn/head/metrics/recall/true_positives/MulMul.dnn/head/metrics/recall/true_positives/ToFloatdnn/head/metrics/recall/ConstV^dnn/head/metrics/recall/true_positives/assert_rank_in/static_checks_determined_all_ok*'
_output_shapes
:���������*
T0
�
/dnn/head/metrics/recall/true_positives/IdentityIdentity1dnn/head/metrics/recall/true_positives/count/read*
T0*
_output_shapes
: 
}
,dnn/head/metrics/recall/true_positives/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
*dnn/head/metrics/recall/true_positives/SumSum*dnn/head/metrics/recall/true_positives/Mul,dnn/head/metrics/recall/true_positives/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
0dnn/head/metrics/recall/true_positives/AssignAdd	AssignAdd,dnn/head/metrics/recall/true_positives/count*dnn/head/metrics/recall/true_positives/Sum*
_output_shapes
: *
use_locking( *
T0*?
_class5
31loc:@dnn/head/metrics/recall/true_positives/count
q
/dnn/head/metrics/recall/false_negatives/Equal/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 
�
-dnn/head/metrics/recall/false_negatives/EqualEqualdnn/head/metrics/recall/Cast_1/dnn/head/metrics/recall/false_negatives/Equal/y*
T0
*'
_output_shapes
:���������
s
1dnn/head/metrics/recall/false_negatives/Equal_1/yConst*
value	B
 Z *
dtype0
*
_output_shapes
: 
�
/dnn/head/metrics/recall/false_negatives/Equal_1Equaldnn/head/metrics/recall/Cast1dnn/head/metrics/recall/false_negatives/Equal_1/y*
T0
*'
_output_shapes
:���������
�
2dnn/head/metrics/recall/false_negatives/LogicalAnd
LogicalAnd-dnn/head/metrics/recall/false_negatives/Equal/dnn/head/metrics/recall/false_negatives/Equal_1*'
_output_shapes
:���������
^
Vdnn/head/metrics/recall/false_negatives/assert_type/statically_determined_correct_typeNoOp
�
?dnn/head/metrics/recall/false_negatives/count/Initializer/zerosConst*
dtype0*
_output_shapes
: *@
_class6
42loc:@dnn/head/metrics/recall/false_negatives/count*
valueB
 *    
�
-dnn/head/metrics/recall/false_negatives/count
VariableV2*
shared_name *@
_class6
42loc:@dnn/head/metrics/recall/false_negatives/count*
	container *
shape: *
dtype0*
_output_shapes
: 
�
4dnn/head/metrics/recall/false_negatives/count/AssignAssign-dnn/head/metrics/recall/false_negatives/count?dnn/head/metrics/recall/false_negatives/count/Initializer/zeros*
T0*@
_class6
42loc:@dnn/head/metrics/recall/false_negatives/count*
validate_shape(*
_output_shapes
: *
use_locking(
�
2dnn/head/metrics/recall/false_negatives/count/readIdentity-dnn/head/metrics/recall/false_negatives/count*
_output_shapes
: *
T0*@
_class6
42loc:@dnn/head/metrics/recall/false_negatives/count
�
/dnn/head/metrics/recall/false_negatives/ToFloatCast2dnn/head/metrics/recall/false_negatives/LogicalAnd*

SrcT0
*'
_output_shapes
:���������*

DstT0
n
,dnn/head/metrics/recall/false_negatives/RankConst*
dtype0*
_output_shapes
: *
value	B :
}
;dnn/head/metrics/recall/false_negatives/assert_rank_in/rankConst*
value	B : *
dtype0*
_output_shapes
: 

<dnn/head/metrics/recall/false_negatives/assert_rank_in/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
m
ednn/head/metrics/recall/false_negatives/assert_rank_in/assert_type/statically_determined_correct_typeNoOp
o
gdnn/head/metrics/recall/false_negatives/assert_rank_in/assert_type_1/statically_determined_correct_typeNoOp
^
Vdnn/head/metrics/recall/false_negatives/assert_rank_in/static_checks_determined_all_okNoOp
�
+dnn/head/metrics/recall/false_negatives/MulMul/dnn/head/metrics/recall/false_negatives/ToFloatdnn/head/metrics/recall/ConstW^dnn/head/metrics/recall/false_negatives/assert_rank_in/static_checks_determined_all_ok*'
_output_shapes
:���������*
T0
�
0dnn/head/metrics/recall/false_negatives/IdentityIdentity2dnn/head/metrics/recall/false_negatives/count/read*
T0*
_output_shapes
: 
~
-dnn/head/metrics/recall/false_negatives/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
+dnn/head/metrics/recall/false_negatives/SumSum+dnn/head/metrics/recall/false_negatives/Mul-dnn/head/metrics/recall/false_negatives/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
1dnn/head/metrics/recall/false_negatives/AssignAdd	AssignAdd-dnn/head/metrics/recall/false_negatives/count+dnn/head/metrics/recall/false_negatives/Sum*@
_class6
42loc:@dnn/head/metrics/recall/false_negatives/count*
_output_shapes
: *
use_locking( *
T0
�
dnn/head/metrics/recall/addAdd/dnn/head/metrics/recall/true_positives/Identity0dnn/head/metrics/recall/false_negatives/Identity*
T0*
_output_shapes
: 
f
!dnn/head/metrics/recall/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
dnn/head/metrics/recall/GreaterGreaterdnn/head/metrics/recall/add!dnn/head/metrics/recall/Greater/y*
T0*
_output_shapes
: 
�
dnn/head/metrics/recall/add_1Add/dnn/head/metrics/recall/true_positives/Identity0dnn/head/metrics/recall/false_negatives/Identity*
T0*
_output_shapes
: 
�
dnn/head/metrics/recall/divRealDiv/dnn/head/metrics/recall/true_positives/Identitydnn/head/metrics/recall/add_1*
T0*
_output_shapes
: 
d
dnn/head/metrics/recall/value/eConst*
_output_shapes
: *
valueB
 *    *
dtype0
�
dnn/head/metrics/recall/valueSelectdnn/head/metrics/recall/Greaterdnn/head/metrics/recall/divdnn/head/metrics/recall/value/e*
T0*
_output_shapes
: 
�
dnn/head/metrics/recall/add_2Add0dnn/head/metrics/recall/true_positives/AssignAdd1dnn/head/metrics/recall/false_negatives/AssignAdd*
T0*
_output_shapes
: 
h
#dnn/head/metrics/recall/Greater_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
!dnn/head/metrics/recall/Greater_1Greaterdnn/head/metrics/recall/add_2#dnn/head/metrics/recall/Greater_1/y*
_output_shapes
: *
T0
�
dnn/head/metrics/recall/add_3Add0dnn/head/metrics/recall/true_positives/AssignAdd1dnn/head/metrics/recall/false_negatives/AssignAdd*
_output_shapes
: *
T0
�
dnn/head/metrics/recall/div_1RealDiv0dnn/head/metrics/recall/true_positives/AssignAdddnn/head/metrics/recall/add_3*
T0*
_output_shapes
: 
h
#dnn/head/metrics/recall/update_op/eConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
!dnn/head/metrics/recall/update_opSelect!dnn/head/metrics/recall/Greater_1dnn/head/metrics/recall/div_1#dnn/head/metrics/recall/update_op/e*
T0*
_output_shapes
: 

:dnn/head/metrics/prediction/mean/broadcast_weights/weightsConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Udnn/head/metrics/prediction/mean/broadcast_weights/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Tdnn/head/metrics/prediction/mean/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
�
Tdnn/head/metrics/prediction/mean/broadcast_weights/assert_broadcastable/values/shapeShapednn/head/predictions/logistic*
out_type0*
_output_shapes
:*
T0
�
Sdnn/head/metrics/prediction/mean/broadcast_weights/assert_broadcastable/values/rankConst*
_output_shapes
: *
value	B :*
dtype0
k
cdnn/head/metrics/prediction/mean/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
�
Bdnn/head/metrics/prediction/mean/broadcast_weights/ones_like/ShapeShapednn/head/predictions/logisticd^dnn/head/metrics/prediction/mean/broadcast_weights/assert_broadcastable/static_scalar_check_success*
out_type0*
_output_shapes
:*
T0
�
Bdnn/head/metrics/prediction/mean/broadcast_weights/ones_like/ConstConstd^dnn/head/metrics/prediction/mean/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
<dnn/head/metrics/prediction/mean/broadcast_weights/ones_likeFillBdnn/head/metrics/prediction/mean/broadcast_weights/ones_like/ShapeBdnn/head/metrics/prediction/mean/broadcast_weights/ones_like/Const*
T0*

index_type0*'
_output_shapes
:���������
�
2dnn/head/metrics/prediction/mean/broadcast_weightsMul:dnn/head/metrics/prediction/mean/broadcast_weights/weights<dnn/head/metrics/prediction/mean/broadcast_weights/ones_like*
T0*'
_output_shapes
:���������
�
8dnn/head/metrics/prediction/mean/total/Initializer/zerosConst*9
_class/
-+loc:@dnn/head/metrics/prediction/mean/total*
valueB
 *    *
dtype0*
_output_shapes
: 
�
&dnn/head/metrics/prediction/mean/total
VariableV2*
dtype0*
_output_shapes
: *
shared_name *9
_class/
-+loc:@dnn/head/metrics/prediction/mean/total*
	container *
shape: 
�
-dnn/head/metrics/prediction/mean/total/AssignAssign&dnn/head/metrics/prediction/mean/total8dnn/head/metrics/prediction/mean/total/Initializer/zeros*
use_locking(*
T0*9
_class/
-+loc:@dnn/head/metrics/prediction/mean/total*
validate_shape(*
_output_shapes
: 
�
+dnn/head/metrics/prediction/mean/total/readIdentity&dnn/head/metrics/prediction/mean/total*
_output_shapes
: *
T0*9
_class/
-+loc:@dnn/head/metrics/prediction/mean/total
�
8dnn/head/metrics/prediction/mean/count/Initializer/zerosConst*9
_class/
-+loc:@dnn/head/metrics/prediction/mean/count*
valueB
 *    *
dtype0*
_output_shapes
: 
�
&dnn/head/metrics/prediction/mean/count
VariableV2*
shared_name *9
_class/
-+loc:@dnn/head/metrics/prediction/mean/count*
	container *
shape: *
dtype0*
_output_shapes
: 
�
-dnn/head/metrics/prediction/mean/count/AssignAssign&dnn/head/metrics/prediction/mean/count8dnn/head/metrics/prediction/mean/count/Initializer/zeros*
use_locking(*
T0*9
_class/
-+loc:@dnn/head/metrics/prediction/mean/count*
validate_shape(*
_output_shapes
: 
�
+dnn/head/metrics/prediction/mean/count/readIdentity&dnn/head/metrics/prediction/mean/count*
_output_shapes
: *
T0*9
_class/
-+loc:@dnn/head/metrics/prediction/mean/count
�
Wdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/shapeShape2dnn/head/metrics/prediction/mean/broadcast_weights*
_output_shapes
:*
T0*
out_type0
�
Vdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/rankConst*
dtype0*
_output_shapes
: *
value	B :
�
Vdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/shapeShapednn/head/predictions/logistic*
T0*
out_type0*
_output_shapes
:
�
Udnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
�
Udnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalar/xConst*
value	B : *
dtype0*
_output_shapes
: 
�
Sdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalarEqualUdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalar/xVdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/rank*
T0*
_output_shapes
: 
�
_dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/SwitchSwitchSdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalarSdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
�
adnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_tIdentityadnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch:1*
T0
*
_output_shapes
: 
�
adnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_fIdentity_dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch*
T0
*
_output_shapes
: 
�
`dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_idIdentitySdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalar*
_output_shapes
: *
T0

�
adnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1SwitchSdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalar`dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0
*f
_class\
ZXloc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalar*
_output_shapes
: : 
�
dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqual�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1*
_output_shapes
: *
T0
�
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchSwitchUdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/rank`dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*h
_class^
\Zloc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/rank*
_output_shapes
: : 
�
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1SwitchVdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/rank`dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*i
_class_
][loc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/rank*
_output_shapes
: : 
�
ydnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/SwitchSwitchdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: : 
�
{dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_tIdentity{dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1*
T0
*
_output_shapes
: 
�
{dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_fIdentityydnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch*
T0
*
_output_shapes
: 
�
zdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_idIdentitydnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: *
T0

�
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConst|^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
���������*
dtype0*
_output_shapes
: 
�
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDims�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim*
_output_shapes

:*

Tdim0*
T0
�
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchSwitchVdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/shape`dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id* 
_output_shapes
::*
T0*i
_class_
][loc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/shape
�
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1Switch�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switchzdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*i
_class_
][loc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/shape* 
_output_shapes
::
�
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeConst|^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB"      *
dtype0*
_output_shapes
:
�
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConst|^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
�
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFill�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const*
T0*

index_type0*
_output_shapes

:
�
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConst|^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
_output_shapes
: *
value	B :*
dtype0
�
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis*
N*
_output_shapes

:*

Tidx0*
T0
�
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConst|^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
���������*
dtype0*
_output_shapes
: 
�
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDims�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim*
T0*
_output_shapes

:*

Tdim0
�
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchSwitchWdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/shape`dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*j
_class`
^\loc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/shape* 
_output_shapes
::
�
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1Switch�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switchzdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*j
_class`
^\loc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/shape* 
_output_shapes
::
�
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperation�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat*<
_output_shapes*
(:���������:���������:*
set_operationa-b*
validate_indices(*
T0
�
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSize�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1*
T0*
out_type0*
_output_shapes
: 
�
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConst|^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B : *
dtype0*
_output_shapes
: 
�
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqual�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims*
T0*
_output_shapes
: 
�
{dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Switchdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankzdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
_output_shapes
: : *
T0
*�
_class�
��loc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank
�
xdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeMerge{dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims*
_output_shapes
: : *
T0
*
N
�
^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/MergeMergexdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Mergecdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1:1*
T0
*
N*
_output_shapes
: : 
�
Odnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/ConstConst*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
�
Qdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/Const_1Const*
_output_shapes
: *
valueB Bweights.shape=*
dtype0
�
Qdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/Const_2Const*
dtype0*
_output_shapes
: *E
value<B: B4dnn/head/metrics/prediction/mean/broadcast_weights:0
�
Qdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/Const_3Const*
dtype0*
_output_shapes
: *
valueB Bvalues.shape=
�
Qdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/Const_4Const*0
value'B% Bdnn/head/predictions/logistic:0*
dtype0*
_output_shapes
: 
�
Qdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/Const_5Const*
_output_shapes
: *
valueB B
is_scalar=*
dtype0
�
\dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/SwitchSwitch^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: : 
�
^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_tIdentity^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Switch:1*
_output_shapes
: *
T0

�
^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_fIdentity\dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Switch*
_output_shapes
: *
T0

�
]dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_idIdentity^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: *
T0

�
Zdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/NoOpNoOp_^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t
�
hdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependencyIdentity^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t[^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/NoOp*
T0
*q
_classg
ecloc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t*
_output_shapes
: 
�
cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0Const_^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: *8
value/B- B'weights can not be broadcast to values.*
dtype0
�
cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1Const_^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
�
cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2Const_^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*E
value<B: B4dnn/head/metrics/prediction/mean/broadcast_weights:0*
dtype0*
_output_shapes
: 
�
cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4Const_^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
�
cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5Const_^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*0
value'B% Bdnn/head/predictions/logistic:0*
dtype0*
_output_shapes
: 
�
cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7Const_^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
�	
\dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/AssertAssertcdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switchcdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2ednn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5ednn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7ednn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3*
T
2	
*
	summarize
�
cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/SwitchSwitch^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge]dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
_output_shapes
: : *
T0
*q
_classg
ecloc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge
�
ednn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1SwitchWdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/shape]dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0*j
_class`
^\loc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/shape* 
_output_shapes
::
�
ednn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2SwitchVdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/shape]dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0*i
_class_
][loc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/shape* 
_output_shapes
::
�
ednn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3SwitchSdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalar]dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0
*f
_class\
ZXloc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalar*
_output_shapes
: : 
�
jdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency_1Identity^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f]^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert*
_output_shapes
: *
T0
*q
_classg
ecloc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f
�
[dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/MergeMergejdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency_1hdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency*
T0
*
N*
_output_shapes
: : 
�
Ddnn/head/metrics/prediction/mean/broadcast_weights_1/ones_like/ShapeShapednn/head/predictions/logistic\^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Merge*
T0*
out_type0*
_output_shapes
:
�
Ddnn/head/metrics/prediction/mean/broadcast_weights_1/ones_like/ConstConst\^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Merge*
_output_shapes
: *
valueB
 *  �?*
dtype0
�
>dnn/head/metrics/prediction/mean/broadcast_weights_1/ones_likeFillDdnn/head/metrics/prediction/mean/broadcast_weights_1/ones_like/ShapeDdnn/head/metrics/prediction/mean/broadcast_weights_1/ones_like/Const*

index_type0*'
_output_shapes
:���������*
T0
�
4dnn/head/metrics/prediction/mean/broadcast_weights_1Mul2dnn/head/metrics/prediction/mean/broadcast_weights>dnn/head/metrics/prediction/mean/broadcast_weights_1/ones_like*
T0*'
_output_shapes
:���������
�
$dnn/head/metrics/prediction/mean/MulMuldnn/head/predictions/logistic4dnn/head/metrics/prediction/mean/broadcast_weights_1*
T0*'
_output_shapes
:���������
w
&dnn/head/metrics/prediction/mean/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
$dnn/head/metrics/prediction/mean/SumSum4dnn/head/metrics/prediction/mean/broadcast_weights_1&dnn/head/metrics/prediction/mean/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
y
(dnn/head/metrics/prediction/mean/Const_1Const*
valueB"       *
dtype0*
_output_shapes
:
�
&dnn/head/metrics/prediction/mean/Sum_1Sum$dnn/head/metrics/prediction/mean/Mul(dnn/head/metrics/prediction/mean/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
*dnn/head/metrics/prediction/mean/AssignAdd	AssignAdd&dnn/head/metrics/prediction/mean/total&dnn/head/metrics/prediction/mean/Sum_1*9
_class/
-+loc:@dnn/head/metrics/prediction/mean/total*
_output_shapes
: *
use_locking( *
T0
�
,dnn/head/metrics/prediction/mean/AssignAdd_1	AssignAdd&dnn/head/metrics/prediction/mean/count$dnn/head/metrics/prediction/mean/Sum%^dnn/head/metrics/prediction/mean/Mul*
use_locking( *
T0*9
_class/
-+loc:@dnn/head/metrics/prediction/mean/count*
_output_shapes
: 
�
(dnn/head/metrics/prediction/mean/truedivRealDiv+dnn/head/metrics/prediction/mean/total/read+dnn/head/metrics/prediction/mean/count/read*
_output_shapes
: *
T0
p
+dnn/head/metrics/prediction/mean/zeros_likeConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
(dnn/head/metrics/prediction/mean/GreaterGreater+dnn/head/metrics/prediction/mean/count/read+dnn/head/metrics/prediction/mean/zeros_like*
T0*
_output_shapes
: 
�
&dnn/head/metrics/prediction/mean/valueSelect(dnn/head/metrics/prediction/mean/Greater(dnn/head/metrics/prediction/mean/truediv+dnn/head/metrics/prediction/mean/zeros_like*
_output_shapes
: *
T0
�
*dnn/head/metrics/prediction/mean/truediv_1RealDiv*dnn/head/metrics/prediction/mean/AssignAdd,dnn/head/metrics/prediction/mean/AssignAdd_1*
T0*
_output_shapes
: 
r
-dnn/head/metrics/prediction/mean/zeros_like_1Const*
valueB
 *    *
dtype0*
_output_shapes
: 
�
*dnn/head/metrics/prediction/mean/Greater_1Greater,dnn/head/metrics/prediction/mean/AssignAdd_1-dnn/head/metrics/prediction/mean/zeros_like_1*
_output_shapes
: *
T0
�
*dnn/head/metrics/prediction/mean/update_opSelect*dnn/head/metrics/prediction/mean/Greater_1*dnn/head/metrics/prediction/mean/truediv_1-dnn/head/metrics/prediction/mean/zeros_like_1*
_output_shapes
: *
T0
m
(dnn/head/metrics/accuracy_baseline/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
&dnn/head/metrics/accuracy_baseline/subSub(dnn/head/metrics/accuracy_baseline/sub/x!dnn/head/metrics/label/mean/value*
_output_shapes
: *
T0
�
(dnn/head/metrics/accuracy_baseline/valueMaximum!dnn/head/metrics/label/mean/value&dnn/head/metrics/accuracy_baseline/sub*
T0*
_output_shapes
: 
o
*dnn/head/metrics/accuracy_baseline/sub_1/xConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
�
(dnn/head/metrics/accuracy_baseline/sub_1Sub*dnn/head/metrics/accuracy_baseline/sub_1/x%dnn/head/metrics/label/mean/update_op*
T0*
_output_shapes
: 
�
,dnn/head/metrics/accuracy_baseline/update_opMaximum%dnn/head/metrics/label/mean/update_op(dnn/head/metrics/accuracy_baseline/sub_1*
T0*
_output_shapes
: 
s
.dnn/head/metrics/auc/broadcast_weights/weightsConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
�
Idnn/head/metrics/auc/broadcast_weights/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Hdnn/head/metrics/auc/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
�
Hdnn/head/metrics/auc/broadcast_weights/assert_broadcastable/values/shapeShapednn/head/predictions/logistic*
T0*
out_type0*
_output_shapes
:
�
Gdnn/head/metrics/auc/broadcast_weights/assert_broadcastable/values/rankConst*
_output_shapes
: *
value	B :*
dtype0
_
Wdnn/head/metrics/auc/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
�
6dnn/head/metrics/auc/broadcast_weights/ones_like/ShapeShapednn/head/predictions/logisticX^dnn/head/metrics/auc/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
�
6dnn/head/metrics/auc/broadcast_weights/ones_like/ConstConstX^dnn/head/metrics/auc/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
0dnn/head/metrics/auc/broadcast_weights/ones_likeFill6dnn/head/metrics/auc/broadcast_weights/ones_like/Shape6dnn/head/metrics/auc/broadcast_weights/ones_like/Const*
T0*

index_type0*'
_output_shapes
:���������
�
&dnn/head/metrics/auc/broadcast_weightsMul.dnn/head/metrics/auc/broadcast_weights/weights0dnn/head/metrics/auc/broadcast_weights/ones_like*
T0*'
_output_shapes
:���������
`
dnn/head/metrics/auc/Cast/xConst*
_output_shapes
: *
valueB
 *    *
dtype0
�
6dnn/head/metrics/auc/assert_greater_equal/GreaterEqualGreaterEqualdnn/head/predictions/logisticdnn/head/metrics/auc/Cast/x*
T0*'
_output_shapes
:���������
�
/dnn/head/metrics/auc/assert_greater_equal/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
-dnn/head/metrics/auc/assert_greater_equal/AllAll6dnn/head/metrics/auc/assert_greater_equal/GreaterEqual/dnn/head/metrics/auc/assert_greater_equal/Const*
_output_shapes
: *
	keep_dims( *

Tidx0
�
6dnn/head/metrics/auc/assert_greater_equal/Assert/ConstConst*.
value%B# Bpredictions must be in [0, 1]*
dtype0*
_output_shapes
: 
�
8dnn/head/metrics/auc/assert_greater_equal/Assert/Const_1Const*b
valueYBW BQCondition x >= y did not hold element-wise:x (dnn/head/predictions/logistic:0) = *
dtype0*
_output_shapes
: 
�
8dnn/head/metrics/auc/assert_greater_equal/Assert/Const_2Const*5
value,B* B$y (dnn/head/metrics/auc/Cast/x:0) = *
dtype0*
_output_shapes
: 
�
Cdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/SwitchSwitch-dnn/head/metrics/auc/assert_greater_equal/All-dnn/head/metrics/auc/assert_greater_equal/All*
_output_shapes
: : *
T0

�
Ednn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_tIdentityEdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
�
Ednn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_fIdentityCdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Switch*
_output_shapes
: *
T0

�
Ddnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/pred_idIdentity-dnn/head/metrics/auc/assert_greater_equal/All*
_output_shapes
: *
T0

�
Adnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/NoOpNoOpF^dnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_t
�
Odnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/control_dependencyIdentityEdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_tB^dnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*X
_classN
LJloc:@dnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_t*
_output_shapes
: 
�
Jdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/data_0ConstF^dnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_f*.
value%B# Bpredictions must be in [0, 1]*
dtype0*
_output_shapes
: 
�
Jdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/data_1ConstF^dnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_f*b
valueYBW BQCondition x >= y did not hold element-wise:x (dnn/head/predictions/logistic:0) = *
dtype0*
_output_shapes
: 
�
Jdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/data_3ConstF^dnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_f*5
value,B* B$y (dnn/head/metrics/auc/Cast/x:0) = *
dtype0*
_output_shapes
: 
�
Cdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/AssertAssertJdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/SwitchJdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/data_0Jdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/data_1Ldnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/Switch_1Jdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/data_3Ldnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/Switch_2*
T	
2*
	summarize
�
Jdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/SwitchSwitch-dnn/head/metrics/auc/assert_greater_equal/AllDdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/pred_id*@
_class6
42loc:@dnn/head/metrics/auc/assert_greater_equal/All*
_output_shapes
: : *
T0

�
Ldnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/Switch_1Switchdnn/head/predictions/logisticDdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/pred_id*
T0*0
_class&
$"loc:@dnn/head/predictions/logistic*:
_output_shapes(
&:���������:���������
�
Ldnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/Switch_2Switchdnn/head/metrics/auc/Cast/xDdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/pred_id*
T0*.
_class$
" loc:@dnn/head/metrics/auc/Cast/x*
_output_shapes
: : 
�
Qdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/control_dependency_1IdentityEdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_fD^dnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert*
T0
*X
_classN
LJloc:@dnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_f*
_output_shapes
: 
�
Bdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/MergeMergeQdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/control_dependency_1Odnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/control_dependency*
N*
_output_shapes
: : *
T0

b
dnn/head/metrics/auc/Cast_1/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
0dnn/head/metrics/auc/assert_less_equal/LessEqual	LessEqualdnn/head/predictions/logisticdnn/head/metrics/auc/Cast_1/x*'
_output_shapes
:���������*
T0
}
,dnn/head/metrics/auc/assert_less_equal/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
*dnn/head/metrics/auc/assert_less_equal/AllAll0dnn/head/metrics/auc/assert_less_equal/LessEqual,dnn/head/metrics/auc/assert_less_equal/Const*
	keep_dims( *

Tidx0*
_output_shapes
: 
�
3dnn/head/metrics/auc/assert_less_equal/Assert/ConstConst*
dtype0*
_output_shapes
: *.
value%B# Bpredictions must be in [0, 1]
�
5dnn/head/metrics/auc/assert_less_equal/Assert/Const_1Const*b
valueYBW BQCondition x <= y did not hold element-wise:x (dnn/head/predictions/logistic:0) = *
dtype0*
_output_shapes
: 
�
5dnn/head/metrics/auc/assert_less_equal/Assert/Const_2Const*7
value.B, B&y (dnn/head/metrics/auc/Cast_1/x:0) = *
dtype0*
_output_shapes
: 
�
@dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/SwitchSwitch*dnn/head/metrics/auc/assert_less_equal/All*dnn/head/metrics/auc/assert_less_equal/All*
_output_shapes
: : *
T0

�
Bdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_tIdentityBdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
�
Bdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_fIdentity@dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Switch*
T0
*
_output_shapes
: 
�
Adnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/pred_idIdentity*dnn/head/metrics/auc/assert_less_equal/All*
T0
*
_output_shapes
: 
�
>dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/NoOpNoOpC^dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_t
�
Ldnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/control_dependencyIdentityBdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_t?^dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/NoOp*
_output_shapes
: *
T0
*U
_classK
IGloc:@dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_t
�
Gdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/data_0ConstC^dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_f*.
value%B# Bpredictions must be in [0, 1]*
dtype0*
_output_shapes
: 
�
Gdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/data_1ConstC^dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_f*b
valueYBW BQCondition x <= y did not hold element-wise:x (dnn/head/predictions/logistic:0) = *
dtype0*
_output_shapes
: 
�
Gdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/data_3ConstC^dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_f*7
value.B, B&y (dnn/head/metrics/auc/Cast_1/x:0) = *
dtype0*
_output_shapes
: 
�
@dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/AssertAssertGdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/SwitchGdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/data_0Gdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/data_1Idnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/Switch_1Gdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/data_3Idnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/Switch_2*
T	
2*
	summarize
�
Gdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/SwitchSwitch*dnn/head/metrics/auc/assert_less_equal/AllAdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/pred_id*
T0
*=
_class3
1/loc:@dnn/head/metrics/auc/assert_less_equal/All*
_output_shapes
: : 
�
Idnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/Switch_1Switchdnn/head/predictions/logisticAdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/pred_id*
T0*0
_class&
$"loc:@dnn/head/predictions/logistic*:
_output_shapes(
&:���������:���������
�
Idnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/Switch_2Switchdnn/head/metrics/auc/Cast_1/xAdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/pred_id*
T0*0
_class&
$"loc:@dnn/head/metrics/auc/Cast_1/x*
_output_shapes
: : 
�
Ndnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/control_dependency_1IdentityBdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_fA^dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert*
_output_shapes
: *
T0
*U
_classK
IGloc:@dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_f
�
?dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/MergeMergeNdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/control_dependency_1Ldnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/control_dependency*
T0
*
N*
_output_shapes
: : 
�
dnn/head/metrics/auc/Cast_2Castdnn/head/assert_range/IdentityC^dnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Merge@^dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Merge*'
_output_shapes
:���������*

DstT0
*

SrcT0
s
"dnn/head/metrics/auc/Reshape/shapeConst*
valueB"����   *
dtype0*
_output_shapes
:
�
dnn/head/metrics/auc/ReshapeReshapednn/head/predictions/logistic"dnn/head/metrics/auc/Reshape/shape*'
_output_shapes
:���������*
T0*
Tshape0
u
$dnn/head/metrics/auc/Reshape_1/shapeConst*
valueB"   ����*
dtype0*
_output_shapes
:
�
dnn/head/metrics/auc/Reshape_1Reshapednn/head/metrics/auc/Cast_2$dnn/head/metrics/auc/Reshape_1/shape*
T0
*
Tshape0*'
_output_shapes
:���������
v
dnn/head/metrics/auc/ShapeShapednn/head/metrics/auc/Reshape*
_output_shapes
:*
T0*
out_type0
r
(dnn/head/metrics/auc/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
t
*dnn/head/metrics/auc/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
t
*dnn/head/metrics/auc/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
"dnn/head/metrics/auc/strided_sliceStridedSlicednn/head/metrics/auc/Shape(dnn/head/metrics/auc/strided_slice/stack*dnn/head/metrics/auc/strided_slice/stack_1*dnn/head/metrics/auc/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
�
dnn/head/metrics/auc/ConstConst*�
value�B��"���ֳϩ�;ϩ$<��v<ϩ�<C��<���<�=ϩ$=	?9=C�M=}ib=��v=�Ʌ=��=2_�=ϩ�=l��=	?�=���=C��=��=}i�=��=���=�� >��>G�
>�>�9>2_>��>ϩ$>�)>l�.>�4>	?9>Wd>>��C>��H>C�M>��R>�X>.D]>}ib>ˎg>�l>h�q>��v>$|>���>Q7�>�Ʌ>�\�>G�>>��><��>�9�>�̗>2_�>��>���>(�>ϩ�>v<�>ϩ>�a�>l��>��>��>b��>	?�>�ѻ>Wd�>���>���>M�>���>�A�>C��>�f�>���>9��>��>���>.D�>���>}i�>$��>ˎ�>r!�>��>�F�>h��>l�>���>^��>$�>���>�� ?��?Q7?��?��?L?�\?�	?G�
?�8?�?B�?�?�]?<�?��?�9?7�?��?�?2_?��?��?-;?��?�� ?("?{`#?ϩ$?#�%?v<'?ʅ(?�)?q+?�a,?�-?l�.?�=0?�1?g�2?�4?c5?b�6?��7?	?9?]�:?��;?=?Wd>?��??��@?R@B?��C?��D?MF?�eG?��H?H�I?�AK?�L?C�M?�O?�fP?>�Q?��R?�BT?9�U?��V?�X?3hY?��Z?��[?.D]?��^?��_?) a?}ib?вc?$�d?xEf?ˎg?�h?r!j?�jk?�l?m�m?�Fo?�p?h�q?�"s?lt?c�u?��v?
Hx?^�y?��z?$|?Ym}?��~? �?*
dtype0*
_output_shapes	
:�
m
#dnn/head/metrics/auc/ExpandDims/dimConst*
_output_shapes
:*
valueB:*
dtype0
�
dnn/head/metrics/auc/ExpandDims
ExpandDimsdnn/head/metrics/auc/Const#dnn/head/metrics/auc/ExpandDims/dim*
T0*
_output_shapes
:	�*

Tdim0
^
dnn/head/metrics/auc/stack/0Const*
_output_shapes
: *
value	B :*
dtype0
�
dnn/head/metrics/auc/stackPackdnn/head/metrics/auc/stack/0"dnn/head/metrics/auc/strided_slice*

axis *
N*
_output_shapes
:*
T0
�
dnn/head/metrics/auc/TileTilednn/head/metrics/auc/ExpandDimsdnn/head/metrics/auc/stack*
T0*(
_output_shapes
:����������*

Tmultiples0
j
#dnn/head/metrics/auc/transpose/RankRankdnn/head/metrics/auc/Reshape*
T0*
_output_shapes
: 
f
$dnn/head/metrics/auc/transpose/sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
"dnn/head/metrics/auc/transpose/subSub#dnn/head/metrics/auc/transpose/Rank$dnn/head/metrics/auc/transpose/sub/y*
_output_shapes
: *
T0
l
*dnn/head/metrics/auc/transpose/Range/startConst*
value	B : *
dtype0*
_output_shapes
: 
l
*dnn/head/metrics/auc/transpose/Range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
$dnn/head/metrics/auc/transpose/RangeRange*dnn/head/metrics/auc/transpose/Range/start#dnn/head/metrics/auc/transpose/Rank*dnn/head/metrics/auc/transpose/Range/delta*
_output_shapes
:*

Tidx0
�
$dnn/head/metrics/auc/transpose/sub_1Sub"dnn/head/metrics/auc/transpose/sub$dnn/head/metrics/auc/transpose/Range*
_output_shapes
:*
T0
�
dnn/head/metrics/auc/transpose	Transposednn/head/metrics/auc/Reshape$dnn/head/metrics/auc/transpose/sub_1*'
_output_shapes
:���������*
Tperm0*
T0
v
%dnn/head/metrics/auc/Tile_1/multiplesConst*
dtype0*
_output_shapes
:*
valueB"�      
�
dnn/head/metrics/auc/Tile_1Tilednn/head/metrics/auc/transpose%dnn/head/metrics/auc/Tile_1/multiples*(
_output_shapes
:����������*

Tmultiples0*
T0
�
dnn/head/metrics/auc/GreaterGreaterdnn/head/metrics/auc/Tile_1dnn/head/metrics/auc/Tile*(
_output_shapes
:����������*
T0
u
dnn/head/metrics/auc/LogicalNot
LogicalNotdnn/head/metrics/auc/Greater*(
_output_shapes
:����������
v
%dnn/head/metrics/auc/Tile_2/multiplesConst*
dtype0*
_output_shapes
:*
valueB"�      
�
dnn/head/metrics/auc/Tile_2Tilednn/head/metrics/auc/Reshape_1%dnn/head/metrics/auc/Tile_2/multiples*
T0
*(
_output_shapes
:����������*

Tmultiples0
v
!dnn/head/metrics/auc/LogicalNot_1
LogicalNotdnn/head/metrics/auc/Tile_2*(
_output_shapes
:����������
�
Kdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/shapeShape&dnn/head/metrics/auc/broadcast_weights*
_output_shapes
:*
T0*
out_type0
�
Jdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/rankConst*
_output_shapes
: *
value	B :*
dtype0
�
Jdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/shapeShapednn/head/predictions/logistic*
T0*
out_type0*
_output_shapes
:
�
Idnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/rankConst*
_output_shapes
: *
value	B :*
dtype0
�
Idnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalar/xConst*
value	B : *
dtype0*
_output_shapes
: 
�
Gdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalarEqualIdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalar/xJdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/rank*
_output_shapes
: *
T0
�
Sdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/SwitchSwitchGdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalarGdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
�
Udnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_tIdentityUdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch:1*
T0
*
_output_shapes
: 
�
Udnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_fIdentitySdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch*
T0
*
_output_shapes
: 
�
Tdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_idIdentityGdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalar*
_output_shapes
: *
T0

�
Udnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1SwitchGdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalarTdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*Z
_classP
NLloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalar*
_output_shapes
: : *
T0

�
sdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqualzdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch|dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1*
T0*
_output_shapes
: 
�
zdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchSwitchIdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/rankTdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*\
_classR
PNloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/rank*
_output_shapes
: : 
�
|dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1SwitchJdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/rankTdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*]
_classS
QOloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/rank*
_output_shapes
: : 
�
mdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/SwitchSwitchsdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_ranksdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: : *
T0

�
odnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_tIdentityodnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1*
T0
*
_output_shapes
: 
�
odnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_fIdentitymdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch*
T0
*
_output_shapes
: 
�
ndnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_idIdentitysdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: 
�
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConstp^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
���������*
dtype0*
_output_shapes
: 
�
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDims�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim*
_output_shapes

:*

Tdim0*
T0
�
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchSwitchJdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/shapeTdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*]
_classS
QOloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/shape* 
_output_shapes
::
�
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1Switch�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switchndnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*]
_classS
QOloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/shape* 
_output_shapes
::
�
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeConstp^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB"      *
dtype0*
_output_shapes
:
�
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConstp^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
_output_shapes
: *
value	B :*
dtype0
�
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFill�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const*
T0*

index_type0*
_output_shapes

:
�
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConstp^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
�
~dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis*

Tidx0*
T0*
N*
_output_shapes

:
�
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConstp^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
_output_shapes
: *
valueB :
���������
�
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDims�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim*
_output_shapes

:*

Tdim0*
T0
�
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchSwitchKdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/shapeTdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*^
_classT
RPloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/shape* 
_output_shapes
::
�
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1Switch�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switchndnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id* 
_output_shapes
::*
T0*^
_classT
RPloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/shape
�
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperation�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1~dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat*
validate_indices(*
T0*<
_output_shapes*
(:���������:���������:*
set_operationa-b
�
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSize�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1*
_output_shapes
: *
T0*
out_type0
�
ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConstp^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B : *
dtype0*
_output_shapes
: 
�
wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqualydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims*
T0*
_output_shapes
: 
�
odnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Switchsdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankndnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0
*�
_class|
zxloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: : 
�
ldnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeMergeodnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims*
T0
*
N*
_output_shapes
: : 
�
Rdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/MergeMergeldnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeWdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1:1*
T0
*
N*
_output_shapes
: : 
�
Cdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/ConstConst*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
�
Ednn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/Const_1Const*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
�
Ednn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/Const_2Const*
dtype0*
_output_shapes
: *9
value0B. B(dnn/head/metrics/auc/broadcast_weights:0
�
Ednn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/Const_3Const*
dtype0*
_output_shapes
: *
valueB Bvalues.shape=
�
Ednn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/Const_4Const*0
value'B% Bdnn/head/predictions/logistic:0*
dtype0*
_output_shapes
: 
�
Ednn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/Const_5Const*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
�
Pdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/SwitchSwitchRdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/MergeRdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: : 
�
Rdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_tIdentityRdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
�
Rdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_fIdentityPdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Switch*
T0
*
_output_shapes
: 
�
Qdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_idIdentityRdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: *
T0

�
Ndnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/NoOpNoOpS^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t
�
\dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependencyIdentityRdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_tO^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/NoOp*
T0
*e
_class[
YWloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t*
_output_shapes
: 
�
Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0ConstS^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
�
Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1ConstS^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
�
Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2ConstS^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*9
value0B. B(dnn/head/metrics/auc/broadcast_weights:0*
dtype0*
_output_shapes
: 
�
Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4ConstS^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
dtype0*
_output_shapes
: *
valueB Bvalues.shape=
�
Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5ConstS^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: *0
value'B% Bdnn/head/predictions/logistic:0*
dtype0
�
Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7ConstS^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: *
valueB B
is_scalar=*
dtype0
�
Pdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/AssertAssertWdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/SwitchWdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2Ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5Ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7Ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3*
T
2	
*
	summarize
�
Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/SwitchSwitchRdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/MergeQdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0
*e
_class[
YWloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: : 
�
Ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1SwitchKdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/shapeQdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*^
_classT
RPloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/shape* 
_output_shapes
::*
T0
�
Ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2SwitchJdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/shapeQdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0*]
_classS
QOloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/shape* 
_output_shapes
::
�
Ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3SwitchGdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalarQdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0
*Z
_classP
NLloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalar*
_output_shapes
: : 
�
^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency_1IdentityRdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_fQ^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert*
T0
*e
_class[
YWloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: 
�
Odnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/MergeMerge^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency_1\dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency*
N*
_output_shapes
: : *
T0

�
8dnn/head/metrics/auc/broadcast_weights_1/ones_like/ShapeShapednn/head/predictions/logisticP^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Merge*
out_type0*
_output_shapes
:*
T0
�
8dnn/head/metrics/auc/broadcast_weights_1/ones_like/ConstConstP^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Merge*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
2dnn/head/metrics/auc/broadcast_weights_1/ones_likeFill8dnn/head/metrics/auc/broadcast_weights_1/ones_like/Shape8dnn/head/metrics/auc/broadcast_weights_1/ones_like/Const*
T0*

index_type0*'
_output_shapes
:���������
�
(dnn/head/metrics/auc/broadcast_weights_1Mul&dnn/head/metrics/auc/broadcast_weights2dnn/head/metrics/auc/broadcast_weights_1/ones_like*'
_output_shapes
:���������*
T0
u
$dnn/head/metrics/auc/Reshape_2/shapeConst*
valueB"   ����*
dtype0*
_output_shapes
:
�
dnn/head/metrics/auc/Reshape_2Reshape(dnn/head/metrics/auc/broadcast_weights_1$dnn/head/metrics/auc/Reshape_2/shape*'
_output_shapes
:���������*
T0*
Tshape0
v
%dnn/head/metrics/auc/Tile_3/multiplesConst*
valueB"�      *
dtype0*
_output_shapes
:
�
dnn/head/metrics/auc/Tile_3Tilednn/head/metrics/auc/Reshape_2%dnn/head/metrics/auc/Tile_3/multiples*(
_output_shapes
:����������*

Tmultiples0*
T0
�
5dnn/head/metrics/auc/true_positives/Initializer/zerosConst*6
_class,
*(loc:@dnn/head/metrics/auc/true_positives*
valueB�*    *
dtype0*
_output_shapes	
:�
�
#dnn/head/metrics/auc/true_positives
VariableV2*
_output_shapes	
:�*
shared_name *6
_class,
*(loc:@dnn/head/metrics/auc/true_positives*
	container *
shape:�*
dtype0
�
*dnn/head/metrics/auc/true_positives/AssignAssign#dnn/head/metrics/auc/true_positives5dnn/head/metrics/auc/true_positives/Initializer/zeros*
T0*6
_class,
*(loc:@dnn/head/metrics/auc/true_positives*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
(dnn/head/metrics/auc/true_positives/readIdentity#dnn/head/metrics/auc/true_positives*
_output_shapes	
:�*
T0*6
_class,
*(loc:@dnn/head/metrics/auc/true_positives
�
dnn/head/metrics/auc/LogicalAnd
LogicalAnddnn/head/metrics/auc/Tile_2dnn/head/metrics/auc/Greater*(
_output_shapes
:����������
�
dnn/head/metrics/auc/ToFloat_2Castdnn/head/metrics/auc/LogicalAnd*

SrcT0
*(
_output_shapes
:����������*

DstT0
�
dnn/head/metrics/auc/mulMuldnn/head/metrics/auc/ToFloat_2dnn/head/metrics/auc/Tile_3*
T0*(
_output_shapes
:����������
l
*dnn/head/metrics/auc/Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
�
dnn/head/metrics/auc/SumSumdnn/head/metrics/auc/mul*dnn/head/metrics/auc/Sum/reduction_indices*
_output_shapes	
:�*
	keep_dims( *

Tidx0*
T0
�
dnn/head/metrics/auc/AssignAdd	AssignAdd#dnn/head/metrics/auc/true_positivesdnn/head/metrics/auc/Sum*
T0*6
_class,
*(loc:@dnn/head/metrics/auc/true_positives*
_output_shapes	
:�*
use_locking( 
�
6dnn/head/metrics/auc/false_negatives/Initializer/zerosConst*
_output_shapes	
:�*7
_class-
+)loc:@dnn/head/metrics/auc/false_negatives*
valueB�*    *
dtype0
�
$dnn/head/metrics/auc/false_negatives
VariableV2*
shared_name *7
_class-
+)loc:@dnn/head/metrics/auc/false_negatives*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
+dnn/head/metrics/auc/false_negatives/AssignAssign$dnn/head/metrics/auc/false_negatives6dnn/head/metrics/auc/false_negatives/Initializer/zeros*
use_locking(*
T0*7
_class-
+)loc:@dnn/head/metrics/auc/false_negatives*
validate_shape(*
_output_shapes	
:�
�
)dnn/head/metrics/auc/false_negatives/readIdentity$dnn/head/metrics/auc/false_negatives*
T0*7
_class-
+)loc:@dnn/head/metrics/auc/false_negatives*
_output_shapes	
:�
�
!dnn/head/metrics/auc/LogicalAnd_1
LogicalAnddnn/head/metrics/auc/Tile_2dnn/head/metrics/auc/LogicalNot*(
_output_shapes
:����������
�
dnn/head/metrics/auc/ToFloat_3Cast!dnn/head/metrics/auc/LogicalAnd_1*

SrcT0
*(
_output_shapes
:����������*

DstT0
�
dnn/head/metrics/auc/mul_1Muldnn/head/metrics/auc/ToFloat_3dnn/head/metrics/auc/Tile_3*
T0*(
_output_shapes
:����������
n
,dnn/head/metrics/auc/Sum_1/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
�
dnn/head/metrics/auc/Sum_1Sumdnn/head/metrics/auc/mul_1,dnn/head/metrics/auc/Sum_1/reduction_indices*
	keep_dims( *

Tidx0*
T0*
_output_shapes	
:�
�
 dnn/head/metrics/auc/AssignAdd_1	AssignAdd$dnn/head/metrics/auc/false_negativesdnn/head/metrics/auc/Sum_1*7
_class-
+)loc:@dnn/head/metrics/auc/false_negatives*
_output_shapes	
:�*
use_locking( *
T0
�
5dnn/head/metrics/auc/true_negatives/Initializer/zerosConst*6
_class,
*(loc:@dnn/head/metrics/auc/true_negatives*
valueB�*    *
dtype0*
_output_shapes	
:�
�
#dnn/head/metrics/auc/true_negatives
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *6
_class,
*(loc:@dnn/head/metrics/auc/true_negatives
�
*dnn/head/metrics/auc/true_negatives/AssignAssign#dnn/head/metrics/auc/true_negatives5dnn/head/metrics/auc/true_negatives/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@dnn/head/metrics/auc/true_negatives*
validate_shape(*
_output_shapes	
:�
�
(dnn/head/metrics/auc/true_negatives/readIdentity#dnn/head/metrics/auc/true_negatives*
T0*6
_class,
*(loc:@dnn/head/metrics/auc/true_negatives*
_output_shapes	
:�
�
!dnn/head/metrics/auc/LogicalAnd_2
LogicalAnd!dnn/head/metrics/auc/LogicalNot_1dnn/head/metrics/auc/LogicalNot*(
_output_shapes
:����������
�
dnn/head/metrics/auc/ToFloat_4Cast!dnn/head/metrics/auc/LogicalAnd_2*

SrcT0
*(
_output_shapes
:����������*

DstT0
�
dnn/head/metrics/auc/mul_2Muldnn/head/metrics/auc/ToFloat_4dnn/head/metrics/auc/Tile_3*
T0*(
_output_shapes
:����������
n
,dnn/head/metrics/auc/Sum_2/reduction_indicesConst*
_output_shapes
: *
value	B :*
dtype0
�
dnn/head/metrics/auc/Sum_2Sumdnn/head/metrics/auc/mul_2,dnn/head/metrics/auc/Sum_2/reduction_indices*
T0*
_output_shapes	
:�*
	keep_dims( *

Tidx0
�
 dnn/head/metrics/auc/AssignAdd_2	AssignAdd#dnn/head/metrics/auc/true_negativesdnn/head/metrics/auc/Sum_2*
_output_shapes	
:�*
use_locking( *
T0*6
_class,
*(loc:@dnn/head/metrics/auc/true_negatives
�
6dnn/head/metrics/auc/false_positives/Initializer/zerosConst*7
_class-
+)loc:@dnn/head/metrics/auc/false_positives*
valueB�*    *
dtype0*
_output_shapes	
:�
�
$dnn/head/metrics/auc/false_positives
VariableV2*
shared_name *7
_class-
+)loc:@dnn/head/metrics/auc/false_positives*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
+dnn/head/metrics/auc/false_positives/AssignAssign$dnn/head/metrics/auc/false_positives6dnn/head/metrics/auc/false_positives/Initializer/zeros*
use_locking(*
T0*7
_class-
+)loc:@dnn/head/metrics/auc/false_positives*
validate_shape(*
_output_shapes	
:�
�
)dnn/head/metrics/auc/false_positives/readIdentity$dnn/head/metrics/auc/false_positives*
T0*7
_class-
+)loc:@dnn/head/metrics/auc/false_positives*
_output_shapes	
:�
�
!dnn/head/metrics/auc/LogicalAnd_3
LogicalAnd!dnn/head/metrics/auc/LogicalNot_1dnn/head/metrics/auc/Greater*(
_output_shapes
:����������
�
dnn/head/metrics/auc/ToFloat_5Cast!dnn/head/metrics/auc/LogicalAnd_3*

SrcT0
*(
_output_shapes
:����������*

DstT0
�
dnn/head/metrics/auc/mul_3Muldnn/head/metrics/auc/ToFloat_5dnn/head/metrics/auc/Tile_3*
T0*(
_output_shapes
:����������
n
,dnn/head/metrics/auc/Sum_3/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
�
dnn/head/metrics/auc/Sum_3Sumdnn/head/metrics/auc/mul_3,dnn/head/metrics/auc/Sum_3/reduction_indices*
T0*
_output_shapes	
:�*
	keep_dims( *

Tidx0
�
 dnn/head/metrics/auc/AssignAdd_3	AssignAdd$dnn/head/metrics/auc/false_positivesdnn/head/metrics/auc/Sum_3*
use_locking( *
T0*7
_class-
+)loc:@dnn/head/metrics/auc/false_positives*
_output_shapes	
:�
_
dnn/head/metrics/auc/add/yConst*
valueB
 *�7�5*
dtype0*
_output_shapes
: 
�
dnn/head/metrics/auc/addAdd(dnn/head/metrics/auc/true_positives/readdnn/head/metrics/auc/add/y*
_output_shapes	
:�*
T0
�
dnn/head/metrics/auc/add_1Add(dnn/head/metrics/auc/true_positives/read)dnn/head/metrics/auc/false_negatives/read*
_output_shapes	
:�*
T0
a
dnn/head/metrics/auc/add_2/yConst*
_output_shapes
: *
valueB
 *�7�5*
dtype0
�
dnn/head/metrics/auc/add_2Adddnn/head/metrics/auc/add_1dnn/head/metrics/auc/add_2/y*
_output_shapes	
:�*
T0

dnn/head/metrics/auc/divRealDivdnn/head/metrics/auc/adddnn/head/metrics/auc/add_2*
T0*
_output_shapes	
:�
�
dnn/head/metrics/auc/add_3Add)dnn/head/metrics/auc/false_positives/read(dnn/head/metrics/auc/true_negatives/read*
T0*
_output_shapes	
:�
a
dnn/head/metrics/auc/add_4/yConst*
dtype0*
_output_shapes
: *
valueB
 *�7�5
�
dnn/head/metrics/auc/add_4Adddnn/head/metrics/auc/add_3dnn/head/metrics/auc/add_4/y*
T0*
_output_shapes	
:�
�
dnn/head/metrics/auc/div_1RealDiv)dnn/head/metrics/auc/false_positives/readdnn/head/metrics/auc/add_4*
T0*
_output_shapes	
:�
t
*dnn/head/metrics/auc/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
w
,dnn/head/metrics/auc/strided_slice_1/stack_1Const*
valueB:�*
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_1/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
�
$dnn/head/metrics/auc/strided_slice_1StridedSlicednn/head/metrics/auc/div_1*dnn/head/metrics/auc/strided_slice_1/stack,dnn/head/metrics/auc/strided_slice_1/stack_1,dnn/head/metrics/auc/strided_slice_1/stack_2*
new_axis_mask *
end_mask *
_output_shapes	
:�*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask 
t
*dnn/head/metrics/auc/strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB:
v
,dnn/head/metrics/auc/strided_slice_2/stack_1Const*
_output_shapes
:*
valueB: *
dtype0
v
,dnn/head/metrics/auc/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
$dnn/head/metrics/auc/strided_slice_2StridedSlicednn/head/metrics/auc/div_1*dnn/head/metrics/auc/strided_slice_2/stack,dnn/head/metrics/auc/strided_slice_2/stack_1,dnn/head/metrics/auc/strided_slice_2/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes	
:�*
T0*
Index0
�
dnn/head/metrics/auc/subSub$dnn/head/metrics/auc/strided_slice_1$dnn/head/metrics/auc/strided_slice_2*
T0*
_output_shapes	
:�
t
*dnn/head/metrics/auc/strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:
w
,dnn/head/metrics/auc/strided_slice_3/stack_1Const*
valueB:�*
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
$dnn/head/metrics/auc/strided_slice_3StridedSlicednn/head/metrics/auc/div*dnn/head/metrics/auc/strided_slice_3/stack,dnn/head/metrics/auc/strided_slice_3/stack_1,dnn/head/metrics/auc/strided_slice_3/stack_2*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask *
_output_shapes	
:�
t
*dnn/head/metrics/auc/strided_slice_4/stackConst*
valueB:*
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_4/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_4/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
$dnn/head/metrics/auc/strided_slice_4StridedSlicednn/head/metrics/auc/div*dnn/head/metrics/auc/strided_slice_4/stack,dnn/head/metrics/auc/strided_slice_4/stack_1,dnn/head/metrics/auc/strided_slice_4/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask*
_output_shapes	
:�*
Index0*
T0*
shrink_axis_mask 
�
dnn/head/metrics/auc/add_5Add$dnn/head/metrics/auc/strided_slice_3$dnn/head/metrics/auc/strided_slice_4*
T0*
_output_shapes	
:�
c
dnn/head/metrics/auc/truediv/yConst*
_output_shapes
: *
valueB
 *   @*
dtype0
�
dnn/head/metrics/auc/truedivRealDivdnn/head/metrics/auc/add_5dnn/head/metrics/auc/truediv/y*
_output_shapes	
:�*
T0
}
dnn/head/metrics/auc/MulMuldnn/head/metrics/auc/subdnn/head/metrics/auc/truediv*
_output_shapes	
:�*
T0
f
dnn/head/metrics/auc/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
dnn/head/metrics/auc/valueSumdnn/head/metrics/auc/Muldnn/head/metrics/auc/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
a
dnn/head/metrics/auc/add_6/yConst*
valueB
 *�7�5*
dtype0*
_output_shapes
: 
�
dnn/head/metrics/auc/add_6Adddnn/head/metrics/auc/AssignAdddnn/head/metrics/auc/add_6/y*
T0*
_output_shapes	
:�
�
dnn/head/metrics/auc/add_7Adddnn/head/metrics/auc/AssignAdd dnn/head/metrics/auc/AssignAdd_1*
_output_shapes	
:�*
T0
a
dnn/head/metrics/auc/add_8/yConst*
valueB
 *�7�5*
dtype0*
_output_shapes
: 
�
dnn/head/metrics/auc/add_8Adddnn/head/metrics/auc/add_7dnn/head/metrics/auc/add_8/y*
T0*
_output_shapes	
:�
�
dnn/head/metrics/auc/div_2RealDivdnn/head/metrics/auc/add_6dnn/head/metrics/auc/add_8*
T0*
_output_shapes	
:�
�
dnn/head/metrics/auc/add_9Add dnn/head/metrics/auc/AssignAdd_3 dnn/head/metrics/auc/AssignAdd_2*
_output_shapes	
:�*
T0
b
dnn/head/metrics/auc/add_10/yConst*
_output_shapes
: *
valueB
 *�7�5*
dtype0
�
dnn/head/metrics/auc/add_10Adddnn/head/metrics/auc/add_9dnn/head/metrics/auc/add_10/y*
_output_shapes	
:�*
T0
�
dnn/head/metrics/auc/div_3RealDiv dnn/head/metrics/auc/AssignAdd_3dnn/head/metrics/auc/add_10*
_output_shapes	
:�*
T0
t
*dnn/head/metrics/auc/strided_slice_5/stackConst*
valueB: *
dtype0*
_output_shapes
:
w
,dnn/head/metrics/auc/strided_slice_5/stack_1Const*
valueB:�*
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_5/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
$dnn/head/metrics/auc/strided_slice_5StridedSlicednn/head/metrics/auc/div_3*dnn/head/metrics/auc/strided_slice_5/stack,dnn/head/metrics/auc/strided_slice_5/stack_1,dnn/head/metrics/auc/strided_slice_5/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes	
:�
t
*dnn/head/metrics/auc/strided_slice_6/stackConst*
valueB:*
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_6/stack_1Const*
_output_shapes
:*
valueB: *
dtype0
v
,dnn/head/metrics/auc/strided_slice_6/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
$dnn/head/metrics/auc/strided_slice_6StridedSlicednn/head/metrics/auc/div_3*dnn/head/metrics/auc/strided_slice_6/stack,dnn/head/metrics/auc/strided_slice_6/stack_1,dnn/head/metrics/auc/strided_slice_6/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes	
:�*
T0*
Index0
�
dnn/head/metrics/auc/sub_1Sub$dnn/head/metrics/auc/strided_slice_5$dnn/head/metrics/auc/strided_slice_6*
T0*
_output_shapes	
:�
t
*dnn/head/metrics/auc/strided_slice_7/stackConst*
valueB: *
dtype0*
_output_shapes
:
w
,dnn/head/metrics/auc/strided_slice_7/stack_1Const*
valueB:�*
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_7/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
$dnn/head/metrics/auc/strided_slice_7StridedSlicednn/head/metrics/auc/div_2*dnn/head/metrics/auc/strided_slice_7/stack,dnn/head/metrics/auc/strided_slice_7/stack_1,dnn/head/metrics/auc/strided_slice_7/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes	
:�
t
*dnn/head/metrics/auc/strided_slice_8/stackConst*
valueB:*
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_8/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_8/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
$dnn/head/metrics/auc/strided_slice_8StridedSlicednn/head/metrics/auc/div_2*dnn/head/metrics/auc/strided_slice_8/stack,dnn/head/metrics/auc/strided_slice_8/stack_1,dnn/head/metrics/auc/strided_slice_8/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes	
:�*
T0*
Index0
�
dnn/head/metrics/auc/add_11Add$dnn/head/metrics/auc/strided_slice_7$dnn/head/metrics/auc/strided_slice_8*
T0*
_output_shapes	
:�
e
 dnn/head/metrics/auc/truediv_1/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
dnn/head/metrics/auc/truediv_1RealDivdnn/head/metrics/auc/add_11 dnn/head/metrics/auc/truediv_1/y*
T0*
_output_shapes	
:�
�
dnn/head/metrics/auc/Mul_1Muldnn/head/metrics/auc/sub_1dnn/head/metrics/auc/truediv_1*
_output_shapes	
:�*
T0
f
dnn/head/metrics/auc/Const_2Const*
valueB: *
dtype0*
_output_shapes
:
�
dnn/head/metrics/auc/update_opSumdnn/head/metrics/auc/Mul_1dnn/head/metrics/auc/Const_2*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
?dnn/head/metrics/auc_precision_recall/broadcast_weights/weightsConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Zdnn/head/metrics/auc_precision_recall/broadcast_weights/assert_broadcastable/weights/shapeConst*
dtype0*
_output_shapes
: *
valueB 
�
Ydnn/head/metrics/auc_precision_recall/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
�
Ydnn/head/metrics/auc_precision_recall/broadcast_weights/assert_broadcastable/values/shapeShapednn/head/predictions/logistic*
T0*
out_type0*
_output_shapes
:
�
Xdnn/head/metrics/auc_precision_recall/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
p
hdnn/head/metrics/auc_precision_recall/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
�
Gdnn/head/metrics/auc_precision_recall/broadcast_weights/ones_like/ShapeShapednn/head/predictions/logistici^dnn/head/metrics/auc_precision_recall/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
�
Gdnn/head/metrics/auc_precision_recall/broadcast_weights/ones_like/ConstConsti^dnn/head/metrics/auc_precision_recall/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Adnn/head/metrics/auc_precision_recall/broadcast_weights/ones_likeFillGdnn/head/metrics/auc_precision_recall/broadcast_weights/ones_like/ShapeGdnn/head/metrics/auc_precision_recall/broadcast_weights/ones_like/Const*'
_output_shapes
:���������*
T0*

index_type0
�
7dnn/head/metrics/auc_precision_recall/broadcast_weightsMul?dnn/head/metrics/auc_precision_recall/broadcast_weights/weightsAdnn/head/metrics/auc_precision_recall/broadcast_weights/ones_like*
T0*'
_output_shapes
:���������
q
,dnn/head/metrics/auc_precision_recall/Cast/xConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
Gdnn/head/metrics/auc_precision_recall/assert_greater_equal/GreaterEqualGreaterEqualdnn/head/predictions/logistic,dnn/head/metrics/auc_precision_recall/Cast/x*
T0*'
_output_shapes
:���������
�
@dnn/head/metrics/auc_precision_recall/assert_greater_equal/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
>dnn/head/metrics/auc_precision_recall/assert_greater_equal/AllAllGdnn/head/metrics/auc_precision_recall/assert_greater_equal/GreaterEqual@dnn/head/metrics/auc_precision_recall/assert_greater_equal/Const*
	keep_dims( *

Tidx0*
_output_shapes
: 
�
Gdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/ConstConst*.
value%B# Bpredictions must be in [0, 1]*
dtype0*
_output_shapes
: 
�
Idnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/Const_1Const*
dtype0*
_output_shapes
: *b
valueYBW BQCondition x >= y did not hold element-wise:x (dnn/head/predictions/logistic:0) = 
�
Idnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/Const_2Const*F
value=B; B5y (dnn/head/metrics/auc_precision_recall/Cast/x:0) = *
dtype0*
_output_shapes
: 
�
Tdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/SwitchSwitch>dnn/head/metrics/auc_precision_recall/assert_greater_equal/All>dnn/head/metrics/auc_precision_recall/assert_greater_equal/All*
_output_shapes
: : *
T0

�
Vdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_tIdentityVdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
�
Vdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_fIdentityTdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Switch*
T0
*
_output_shapes
: 
�
Udnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/pred_idIdentity>dnn/head/metrics/auc_precision_recall/assert_greater_equal/All*
T0
*
_output_shapes
: 
�
Rdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/NoOpNoOpW^dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_t
�
`dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/control_dependencyIdentityVdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_tS^dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*i
_class_
][loc:@dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_t*
_output_shapes
: 
�
[dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/data_0ConstW^dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_f*.
value%B# Bpredictions must be in [0, 1]*
dtype0*
_output_shapes
: 
�
[dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/data_1ConstW^dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_f*b
valueYBW BQCondition x >= y did not hold element-wise:x (dnn/head/predictions/logistic:0) = *
dtype0*
_output_shapes
: 
�
[dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/data_3ConstW^dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_f*
_output_shapes
: *F
value=B; B5y (dnn/head/metrics/auc_precision_recall/Cast/x:0) = *
dtype0
�
Tdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/AssertAssert[dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/Switch[dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/data_0[dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/data_1]dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/Switch_1[dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/data_3]dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/Switch_2*
T	
2*
	summarize
�
[dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/SwitchSwitch>dnn/head/metrics/auc_precision_recall/assert_greater_equal/AllUdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/pred_id*
T0
*Q
_classG
ECloc:@dnn/head/metrics/auc_precision_recall/assert_greater_equal/All*
_output_shapes
: : 
�
]dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/Switch_1Switchdnn/head/predictions/logisticUdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/pred_id*
T0*0
_class&
$"loc:@dnn/head/predictions/logistic*:
_output_shapes(
&:���������:���������
�
]dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/Switch_2Switch,dnn/head/metrics/auc_precision_recall/Cast/xUdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/pred_id*
T0*?
_class5
31loc:@dnn/head/metrics/auc_precision_recall/Cast/x*
_output_shapes
: : 
�
bdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/control_dependency_1IdentityVdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_fU^dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert*
T0
*i
_class_
][loc:@dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_f*
_output_shapes
: 
�
Sdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/MergeMergebdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/control_dependency_1`dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/control_dependency*
T0
*
N*
_output_shapes
: : 
s
.dnn/head/metrics/auc_precision_recall/Cast_1/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Adnn/head/metrics/auc_precision_recall/assert_less_equal/LessEqual	LessEqualdnn/head/predictions/logistic.dnn/head/metrics/auc_precision_recall/Cast_1/x*
T0*'
_output_shapes
:���������
�
=dnn/head/metrics/auc_precision_recall/assert_less_equal/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
;dnn/head/metrics/auc_precision_recall/assert_less_equal/AllAllAdnn/head/metrics/auc_precision_recall/assert_less_equal/LessEqual=dnn/head/metrics/auc_precision_recall/assert_less_equal/Const*
_output_shapes
: *
	keep_dims( *

Tidx0
�
Ddnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/ConstConst*.
value%B# Bpredictions must be in [0, 1]*
dtype0*
_output_shapes
: 
�
Fdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/Const_1Const*
dtype0*
_output_shapes
: *b
valueYBW BQCondition x <= y did not hold element-wise:x (dnn/head/predictions/logistic:0) = 
�
Fdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/Const_2Const*H
value?B= B7y (dnn/head/metrics/auc_precision_recall/Cast_1/x:0) = *
dtype0*
_output_shapes
: 
�
Qdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/SwitchSwitch;dnn/head/metrics/auc_precision_recall/assert_less_equal/All;dnn/head/metrics/auc_precision_recall/assert_less_equal/All*
_output_shapes
: : *
T0

�
Sdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_tIdentitySdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Switch:1*
_output_shapes
: *
T0

�
Sdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_fIdentityQdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Switch*
_output_shapes
: *
T0

�
Rdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/pred_idIdentity;dnn/head/metrics/auc_precision_recall/assert_less_equal/All*
T0
*
_output_shapes
: 
�
Odnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/NoOpNoOpT^dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_t
�
]dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/control_dependencyIdentitySdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_tP^dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/NoOp*
T0
*f
_class\
ZXloc:@dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_t*
_output_shapes
: 
�
Xdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/data_0ConstT^dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_f*.
value%B# Bpredictions must be in [0, 1]*
dtype0*
_output_shapes
: 
�
Xdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/data_1ConstT^dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_f*b
valueYBW BQCondition x <= y did not hold element-wise:x (dnn/head/predictions/logistic:0) = *
dtype0*
_output_shapes
: 
�
Xdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/data_3ConstT^dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_f*H
value?B= B7y (dnn/head/metrics/auc_precision_recall/Cast_1/x:0) = *
dtype0*
_output_shapes
: 
�
Qdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/AssertAssertXdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/SwitchXdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/data_0Xdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/data_1Zdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/Switch_1Xdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/data_3Zdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/Switch_2*
	summarize*
T	
2
�
Xdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/SwitchSwitch;dnn/head/metrics/auc_precision_recall/assert_less_equal/AllRdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/pred_id*
T0
*N
_classD
B@loc:@dnn/head/metrics/auc_precision_recall/assert_less_equal/All*
_output_shapes
: : 
�
Zdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/Switch_1Switchdnn/head/predictions/logisticRdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/pred_id*
T0*0
_class&
$"loc:@dnn/head/predictions/logistic*:
_output_shapes(
&:���������:���������
�
Zdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/Switch_2Switch.dnn/head/metrics/auc_precision_recall/Cast_1/xRdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/pred_id*
_output_shapes
: : *
T0*A
_class7
53loc:@dnn/head/metrics/auc_precision_recall/Cast_1/x
�
_dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/control_dependency_1IdentitySdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_fR^dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert*
T0
*f
_class\
ZXloc:@dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_f*
_output_shapes
: 
�
Pdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/MergeMerge_dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/control_dependency_1]dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/control_dependency*
N*
_output_shapes
: : *
T0

�
,dnn/head/metrics/auc_precision_recall/Cast_2Castdnn/head/assert_range/IdentityT^dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/MergeQ^dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Merge*'
_output_shapes
:���������*

DstT0
*

SrcT0
�
3dnn/head/metrics/auc_precision_recall/Reshape/shapeConst*
valueB"����   *
dtype0*
_output_shapes
:
�
-dnn/head/metrics/auc_precision_recall/ReshapeReshapednn/head/predictions/logistic3dnn/head/metrics/auc_precision_recall/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:���������
�
5dnn/head/metrics/auc_precision_recall/Reshape_1/shapeConst*
valueB"   ����*
dtype0*
_output_shapes
:
�
/dnn/head/metrics/auc_precision_recall/Reshape_1Reshape,dnn/head/metrics/auc_precision_recall/Cast_25dnn/head/metrics/auc_precision_recall/Reshape_1/shape*
Tshape0*'
_output_shapes
:���������*
T0

�
+dnn/head/metrics/auc_precision_recall/ShapeShape-dnn/head/metrics/auc_precision_recall/Reshape*
T0*
out_type0*
_output_shapes
:
�
9dnn/head/metrics/auc_precision_recall/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
;dnn/head/metrics/auc_precision_recall/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
;dnn/head/metrics/auc_precision_recall/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
3dnn/head/metrics/auc_precision_recall/strided_sliceStridedSlice+dnn/head/metrics/auc_precision_recall/Shape9dnn/head/metrics/auc_precision_recall/strided_slice/stack;dnn/head/metrics/auc_precision_recall/strided_slice/stack_1;dnn/head/metrics/auc_precision_recall/strided_slice/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask
�
+dnn/head/metrics/auc_precision_recall/ConstConst*�
value�B��"���ֳϩ�;ϩ$<��v<ϩ�<C��<���<�=ϩ$=	?9=C�M=}ib=��v=�Ʌ=��=2_�=ϩ�=l��=	?�=���=C��=��=}i�=��=���=�� >��>G�
>�>�9>2_>��>ϩ$>�)>l�.>�4>	?9>Wd>>��C>��H>C�M>��R>�X>.D]>}ib>ˎg>�l>h�q>��v>$|>���>Q7�>�Ʌ>�\�>G�>>��><��>�9�>�̗>2_�>��>���>(�>ϩ�>v<�>ϩ>�a�>l��>��>��>b��>	?�>�ѻ>Wd�>���>���>M�>���>�A�>C��>�f�>���>9��>��>���>.D�>���>}i�>$��>ˎ�>r!�>��>�F�>h��>l�>���>^��>$�>���>�� ?��?Q7?��?��?L?�\?�	?G�
?�8?�?B�?�?�]?<�?��?�9?7�?��?�?2_?��?��?-;?��?�� ?("?{`#?ϩ$?#�%?v<'?ʅ(?�)?q+?�a,?�-?l�.?�=0?�1?g�2?�4?c5?b�6?��7?	?9?]�:?��;?=?Wd>?��??��@?R@B?��C?��D?MF?�eG?��H?H�I?�AK?�L?C�M?�O?�fP?>�Q?��R?�BT?9�U?��V?�X?3hY?��Z?��[?.D]?��^?��_?) a?}ib?вc?$�d?xEf?ˎg?�h?r!j?�jk?�l?m�m?�Fo?�p?h�q?�"s?lt?c�u?��v?
Hx?^�y?��z?$|?Ym}?��~? �?*
dtype0*
_output_shapes	
:�
~
4dnn/head/metrics/auc_precision_recall/ExpandDims/dimConst*
valueB:*
dtype0*
_output_shapes
:
�
0dnn/head/metrics/auc_precision_recall/ExpandDims
ExpandDims+dnn/head/metrics/auc_precision_recall/Const4dnn/head/metrics/auc_precision_recall/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:	�
o
-dnn/head/metrics/auc_precision_recall/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 
�
+dnn/head/metrics/auc_precision_recall/stackPack-dnn/head/metrics/auc_precision_recall/stack/03dnn/head/metrics/auc_precision_recall/strided_slice*
T0*

axis *
N*
_output_shapes
:
�
*dnn/head/metrics/auc_precision_recall/TileTile0dnn/head/metrics/auc_precision_recall/ExpandDims+dnn/head/metrics/auc_precision_recall/stack*

Tmultiples0*
T0*(
_output_shapes
:����������
�
4dnn/head/metrics/auc_precision_recall/transpose/RankRank-dnn/head/metrics/auc_precision_recall/Reshape*
T0*
_output_shapes
: 
w
5dnn/head/metrics/auc_precision_recall/transpose/sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
3dnn/head/metrics/auc_precision_recall/transpose/subSub4dnn/head/metrics/auc_precision_recall/transpose/Rank5dnn/head/metrics/auc_precision_recall/transpose/sub/y*
_output_shapes
: *
T0
}
;dnn/head/metrics/auc_precision_recall/transpose/Range/startConst*
value	B : *
dtype0*
_output_shapes
: 
}
;dnn/head/metrics/auc_precision_recall/transpose/Range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
5dnn/head/metrics/auc_precision_recall/transpose/RangeRange;dnn/head/metrics/auc_precision_recall/transpose/Range/start4dnn/head/metrics/auc_precision_recall/transpose/Rank;dnn/head/metrics/auc_precision_recall/transpose/Range/delta*

Tidx0*
_output_shapes
:
�
5dnn/head/metrics/auc_precision_recall/transpose/sub_1Sub3dnn/head/metrics/auc_precision_recall/transpose/sub5dnn/head/metrics/auc_precision_recall/transpose/Range*
T0*
_output_shapes
:
�
/dnn/head/metrics/auc_precision_recall/transpose	Transpose-dnn/head/metrics/auc_precision_recall/Reshape5dnn/head/metrics/auc_precision_recall/transpose/sub_1*
T0*'
_output_shapes
:���������*
Tperm0
�
6dnn/head/metrics/auc_precision_recall/Tile_1/multiplesConst*
valueB"�      *
dtype0*
_output_shapes
:
�
,dnn/head/metrics/auc_precision_recall/Tile_1Tile/dnn/head/metrics/auc_precision_recall/transpose6dnn/head/metrics/auc_precision_recall/Tile_1/multiples*(
_output_shapes
:����������*

Tmultiples0*
T0
�
-dnn/head/metrics/auc_precision_recall/GreaterGreater,dnn/head/metrics/auc_precision_recall/Tile_1*dnn/head/metrics/auc_precision_recall/Tile*
T0*(
_output_shapes
:����������
�
0dnn/head/metrics/auc_precision_recall/LogicalNot
LogicalNot-dnn/head/metrics/auc_precision_recall/Greater*(
_output_shapes
:����������
�
6dnn/head/metrics/auc_precision_recall/Tile_2/multiplesConst*
valueB"�      *
dtype0*
_output_shapes
:
�
,dnn/head/metrics/auc_precision_recall/Tile_2Tile/dnn/head/metrics/auc_precision_recall/Reshape_16dnn/head/metrics/auc_precision_recall/Tile_2/multiples*

Tmultiples0*
T0
*(
_output_shapes
:����������
�
2dnn/head/metrics/auc_precision_recall/LogicalNot_1
LogicalNot,dnn/head/metrics/auc_precision_recall/Tile_2*(
_output_shapes
:����������
�
\dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/shapeShape7dnn/head/metrics/auc_precision_recall/broadcast_weights*
_output_shapes
:*
T0*
out_type0
�
[dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/rankConst*
_output_shapes
: *
value	B :*
dtype0
�
[dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/shapeShapednn/head/predictions/logistic*
T0*
out_type0*
_output_shapes
:
�
Zdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
�
Zdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalar/xConst*
value	B : *
dtype0*
_output_shapes
: 
�
Xdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalarEqualZdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalar/x[dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/rank*
_output_shapes
: *
T0
�
ddnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/SwitchSwitchXdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalarXdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
�
fdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_tIdentityfdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch:1*
T0
*
_output_shapes
: 
�
fdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_fIdentityddnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch*
T0
*
_output_shapes
: 
�
ednn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_idIdentityXdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: 
�
fdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1SwitchXdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalarednn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0
*k
_classa
_]loc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalar*
_output_shapes
: : 
�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqual�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1*
T0*
_output_shapes
: 
�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchSwitchZdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/rankednn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*m
_classc
a_loc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/rank*
_output_shapes
: : 
�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1Switch[dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/rankednn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*n
_classd
b`loc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/rank*
_output_shapes
: : 
�
~dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/SwitchSwitch�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: : *
T0

�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_tIdentity�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1*
_output_shapes
: *
T0

�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_fIdentity~dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch*
_output_shapes
: *
T0

�
dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_idIdentity�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: *
T0

�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConst�^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
���������*
dtype0*
_output_shapes
: 
�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDims�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim*

Tdim0*
T0*
_output_shapes

:
�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchSwitch[dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/shapeednn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*n
_classd
b`loc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/shape* 
_output_shapes
::
�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1Switch�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switchdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*n
_classd
b`loc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/shape* 
_output_shapes
::
�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeConst�^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB"      *
dtype0*
_output_shapes
:
�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConst�^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFill�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const*
T0*

index_type0*
_output_shapes

:
�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConst�^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis*
T0*
N*
_output_shapes

:*

Tidx0
�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConst�^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
���������*
dtype0*
_output_shapes
: 
�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDims�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes

:
�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchSwitch\dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/shapeednn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id* 
_output_shapes
::*
T0*o
_classe
caloc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/shape
�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1Switch�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switchdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id* 
_output_shapes
::*
T0*o
_classe
caloc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/shape
�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperation�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat*<
_output_shapes*
(:���������:���������:*
set_operationa-b*
validate_indices(*
T0
�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSize�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1*
T0*
out_type0*
_output_shapes
: 
�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConst�^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B : *
dtype0*
_output_shapes
: 
�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqual�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims*
_output_shapes
: *
T0
�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Switch�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
_output_shapes
: : *
T0
*�
_class�
��loc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank
�
}dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeMerge�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims*
T0
*
N*
_output_shapes
: : 
�
cdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/MergeMerge}dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Mergehdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1:1*
T0
*
N*
_output_shapes
: : 
�
Tdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/ConstConst*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
�
Vdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/Const_1Const*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
�
Vdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/Const_2Const*J
valueAB? B9dnn/head/metrics/auc_precision_recall/broadcast_weights:0*
dtype0*
_output_shapes
: 
�
Vdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/Const_3Const*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
�
Vdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/Const_4Const*0
value'B% Bdnn/head/predictions/logistic:0*
dtype0*
_output_shapes
: 
�
Vdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/Const_5Const*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
�
adnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/SwitchSwitchcdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Mergecdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: : *
T0

�
cdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_tIdentitycdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Switch:1*
_output_shapes
: *
T0

�
cdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_fIdentityadnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Switch*
_output_shapes
: *
T0

�
bdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_idIdentitycdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: 
�
_dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/NoOpNoOpd^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t
�
mdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependencyIdentitycdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t`^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/NoOp*
T0
*v
_classl
jhloc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t*
_output_shapes
: 
�
hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0Constd^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
�
hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1Constd^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
�
hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2Constd^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
dtype0*
_output_shapes
: *J
valueAB? B9dnn/head/metrics/auc_precision_recall/broadcast_weights:0
�
hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4Constd^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
�
hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5Constd^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*0
value'B% Bdnn/head/predictions/logistic:0*
dtype0*
_output_shapes
: 
�
hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7Constd^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
�	
adnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/AssertAsserthdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switchhdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2jdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5jdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7jdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3*
T
2	
*
	summarize
�
hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/SwitchSwitchcdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Mergebdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0
*v
_classl
jhloc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: : 
�
jdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1Switch\dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/shapebdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id* 
_output_shapes
::*
T0*o
_classe
caloc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/shape
�
jdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2Switch[dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/shapebdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0*n
_classd
b`loc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/shape* 
_output_shapes
::
�
jdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3SwitchXdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalarbdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
_output_shapes
: : *
T0
*k
_classa
_]loc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalar
�
odnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency_1Identitycdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_fb^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert*
T0
*v
_classl
jhloc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: 
�
`dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/MergeMergeodnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency_1mdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency*
N*
_output_shapes
: : *
T0

�
Idnn/head/metrics/auc_precision_recall/broadcast_weights_1/ones_like/ShapeShapednn/head/predictions/logistica^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Merge*
_output_shapes
:*
T0*
out_type0
�
Idnn/head/metrics/auc_precision_recall/broadcast_weights_1/ones_like/ConstConsta^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Merge*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Cdnn/head/metrics/auc_precision_recall/broadcast_weights_1/ones_likeFillIdnn/head/metrics/auc_precision_recall/broadcast_weights_1/ones_like/ShapeIdnn/head/metrics/auc_precision_recall/broadcast_weights_1/ones_like/Const*
T0*

index_type0*'
_output_shapes
:���������
�
9dnn/head/metrics/auc_precision_recall/broadcast_weights_1Mul7dnn/head/metrics/auc_precision_recall/broadcast_weightsCdnn/head/metrics/auc_precision_recall/broadcast_weights_1/ones_like*
T0*'
_output_shapes
:���������
�
5dnn/head/metrics/auc_precision_recall/Reshape_2/shapeConst*
_output_shapes
:*
valueB"   ����*
dtype0
�
/dnn/head/metrics/auc_precision_recall/Reshape_2Reshape9dnn/head/metrics/auc_precision_recall/broadcast_weights_15dnn/head/metrics/auc_precision_recall/Reshape_2/shape*
T0*
Tshape0*'
_output_shapes
:���������
�
6dnn/head/metrics/auc_precision_recall/Tile_3/multiplesConst*
valueB"�      *
dtype0*
_output_shapes
:
�
,dnn/head/metrics/auc_precision_recall/Tile_3Tile/dnn/head/metrics/auc_precision_recall/Reshape_26dnn/head/metrics/auc_precision_recall/Tile_3/multiples*(
_output_shapes
:����������*

Tmultiples0*
T0
�
Fdnn/head/metrics/auc_precision_recall/true_positives/Initializer/zerosConst*G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_positives*
valueB�*    *
dtype0*
_output_shapes	
:�
�
4dnn/head/metrics/auc_precision_recall/true_positives
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_positives
�
;dnn/head/metrics/auc_precision_recall/true_positives/AssignAssign4dnn/head/metrics/auc_precision_recall/true_positivesFdnn/head/metrics/auc_precision_recall/true_positives/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_positives*
validate_shape(*
_output_shapes	
:�
�
9dnn/head/metrics/auc_precision_recall/true_positives/readIdentity4dnn/head/metrics/auc_precision_recall/true_positives*
T0*G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_positives*
_output_shapes	
:�
�
0dnn/head/metrics/auc_precision_recall/LogicalAnd
LogicalAnd,dnn/head/metrics/auc_precision_recall/Tile_2-dnn/head/metrics/auc_precision_recall/Greater*(
_output_shapes
:����������
�
/dnn/head/metrics/auc_precision_recall/ToFloat_2Cast0dnn/head/metrics/auc_precision_recall/LogicalAnd*(
_output_shapes
:����������*

DstT0*

SrcT0

�
)dnn/head/metrics/auc_precision_recall/mulMul/dnn/head/metrics/auc_precision_recall/ToFloat_2,dnn/head/metrics/auc_precision_recall/Tile_3*(
_output_shapes
:����������*
T0
}
;dnn/head/metrics/auc_precision_recall/Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
�
)dnn/head/metrics/auc_precision_recall/SumSum)dnn/head/metrics/auc_precision_recall/mul;dnn/head/metrics/auc_precision_recall/Sum/reduction_indices*
T0*
_output_shapes	
:�*
	keep_dims( *

Tidx0
�
/dnn/head/metrics/auc_precision_recall/AssignAdd	AssignAdd4dnn/head/metrics/auc_precision_recall/true_positives)dnn/head/metrics/auc_precision_recall/Sum*
T0*G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_positives*
_output_shapes	
:�*
use_locking( 
�
Gdnn/head/metrics/auc_precision_recall/false_negatives/Initializer/zerosConst*H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_negatives*
valueB�*    *
dtype0*
_output_shapes	
:�
�
5dnn/head/metrics/auc_precision_recall/false_negatives
VariableV2*H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_negatives*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
<dnn/head/metrics/auc_precision_recall/false_negatives/AssignAssign5dnn/head/metrics/auc_precision_recall/false_negativesGdnn/head/metrics/auc_precision_recall/false_negatives/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_negatives
�
:dnn/head/metrics/auc_precision_recall/false_negatives/readIdentity5dnn/head/metrics/auc_precision_recall/false_negatives*
T0*H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_negatives*
_output_shapes	
:�
�
2dnn/head/metrics/auc_precision_recall/LogicalAnd_1
LogicalAnd,dnn/head/metrics/auc_precision_recall/Tile_20dnn/head/metrics/auc_precision_recall/LogicalNot*(
_output_shapes
:����������
�
/dnn/head/metrics/auc_precision_recall/ToFloat_3Cast2dnn/head/metrics/auc_precision_recall/LogicalAnd_1*(
_output_shapes
:����������*

DstT0*

SrcT0

�
+dnn/head/metrics/auc_precision_recall/mul_1Mul/dnn/head/metrics/auc_precision_recall/ToFloat_3,dnn/head/metrics/auc_precision_recall/Tile_3*(
_output_shapes
:����������*
T0

=dnn/head/metrics/auc_precision_recall/Sum_1/reduction_indicesConst*
_output_shapes
: *
value	B :*
dtype0
�
+dnn/head/metrics/auc_precision_recall/Sum_1Sum+dnn/head/metrics/auc_precision_recall/mul_1=dnn/head/metrics/auc_precision_recall/Sum_1/reduction_indices*
	keep_dims( *

Tidx0*
T0*
_output_shapes	
:�
�
1dnn/head/metrics/auc_precision_recall/AssignAdd_1	AssignAdd5dnn/head/metrics/auc_precision_recall/false_negatives+dnn/head/metrics/auc_precision_recall/Sum_1*
use_locking( *
T0*H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_negatives*
_output_shapes	
:�
�
Fdnn/head/metrics/auc_precision_recall/true_negatives/Initializer/zerosConst*G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_negatives*
valueB�*    *
dtype0*
_output_shapes	
:�
�
4dnn/head/metrics/auc_precision_recall/true_negatives
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_negatives*
	container *
shape:�
�
;dnn/head/metrics/auc_precision_recall/true_negatives/AssignAssign4dnn/head/metrics/auc_precision_recall/true_negativesFdnn/head/metrics/auc_precision_recall/true_negatives/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_negatives
�
9dnn/head/metrics/auc_precision_recall/true_negatives/readIdentity4dnn/head/metrics/auc_precision_recall/true_negatives*
T0*G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_negatives*
_output_shapes	
:�
�
2dnn/head/metrics/auc_precision_recall/LogicalAnd_2
LogicalAnd2dnn/head/metrics/auc_precision_recall/LogicalNot_10dnn/head/metrics/auc_precision_recall/LogicalNot*(
_output_shapes
:����������
�
/dnn/head/metrics/auc_precision_recall/ToFloat_4Cast2dnn/head/metrics/auc_precision_recall/LogicalAnd_2*

SrcT0
*(
_output_shapes
:����������*

DstT0
�
+dnn/head/metrics/auc_precision_recall/mul_2Mul/dnn/head/metrics/auc_precision_recall/ToFloat_4,dnn/head/metrics/auc_precision_recall/Tile_3*
T0*(
_output_shapes
:����������

=dnn/head/metrics/auc_precision_recall/Sum_2/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
�
+dnn/head/metrics/auc_precision_recall/Sum_2Sum+dnn/head/metrics/auc_precision_recall/mul_2=dnn/head/metrics/auc_precision_recall/Sum_2/reduction_indices*
	keep_dims( *

Tidx0*
T0*
_output_shapes	
:�
�
1dnn/head/metrics/auc_precision_recall/AssignAdd_2	AssignAdd4dnn/head/metrics/auc_precision_recall/true_negatives+dnn/head/metrics/auc_precision_recall/Sum_2*
use_locking( *
T0*G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_negatives*
_output_shapes	
:�
�
Gdnn/head/metrics/auc_precision_recall/false_positives/Initializer/zerosConst*H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_positives*
valueB�*    *
dtype0*
_output_shapes	
:�
�
5dnn/head/metrics/auc_precision_recall/false_positives
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_positives*
	container *
shape:�
�
<dnn/head/metrics/auc_precision_recall/false_positives/AssignAssign5dnn/head/metrics/auc_precision_recall/false_positivesGdnn/head/metrics/auc_precision_recall/false_positives/Initializer/zeros*H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_positives*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
:dnn/head/metrics/auc_precision_recall/false_positives/readIdentity5dnn/head/metrics/auc_precision_recall/false_positives*
T0*H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_positives*
_output_shapes	
:�
�
2dnn/head/metrics/auc_precision_recall/LogicalAnd_3
LogicalAnd2dnn/head/metrics/auc_precision_recall/LogicalNot_1-dnn/head/metrics/auc_precision_recall/Greater*(
_output_shapes
:����������
�
/dnn/head/metrics/auc_precision_recall/ToFloat_5Cast2dnn/head/metrics/auc_precision_recall/LogicalAnd_3*

SrcT0
*(
_output_shapes
:����������*

DstT0
�
+dnn/head/metrics/auc_precision_recall/mul_3Mul/dnn/head/metrics/auc_precision_recall/ToFloat_5,dnn/head/metrics/auc_precision_recall/Tile_3*
T0*(
_output_shapes
:����������

=dnn/head/metrics/auc_precision_recall/Sum_3/reduction_indicesConst*
_output_shapes
: *
value	B :*
dtype0
�
+dnn/head/metrics/auc_precision_recall/Sum_3Sum+dnn/head/metrics/auc_precision_recall/mul_3=dnn/head/metrics/auc_precision_recall/Sum_3/reduction_indices*
	keep_dims( *

Tidx0*
T0*
_output_shapes	
:�
�
1dnn/head/metrics/auc_precision_recall/AssignAdd_3	AssignAdd5dnn/head/metrics/auc_precision_recall/false_positives+dnn/head/metrics/auc_precision_recall/Sum_3*
use_locking( *
T0*H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_positives*
_output_shapes	
:�
p
+dnn/head/metrics/auc_precision_recall/add/yConst*
valueB
 *�7�5*
dtype0*
_output_shapes
: 
�
)dnn/head/metrics/auc_precision_recall/addAdd9dnn/head/metrics/auc_precision_recall/true_positives/read+dnn/head/metrics/auc_precision_recall/add/y*
T0*
_output_shapes	
:�
�
+dnn/head/metrics/auc_precision_recall/add_1Add9dnn/head/metrics/auc_precision_recall/true_positives/read:dnn/head/metrics/auc_precision_recall/false_negatives/read*
_output_shapes	
:�*
T0
r
-dnn/head/metrics/auc_precision_recall/add_2/yConst*
valueB
 *�7�5*
dtype0*
_output_shapes
: 
�
+dnn/head/metrics/auc_precision_recall/add_2Add+dnn/head/metrics/auc_precision_recall/add_1-dnn/head/metrics/auc_precision_recall/add_2/y*
T0*
_output_shapes	
:�
�
)dnn/head/metrics/auc_precision_recall/divRealDiv)dnn/head/metrics/auc_precision_recall/add+dnn/head/metrics/auc_precision_recall/add_2*
_output_shapes	
:�*
T0
r
-dnn/head/metrics/auc_precision_recall/add_3/yConst*
valueB
 *�7�5*
dtype0*
_output_shapes
: 
�
+dnn/head/metrics/auc_precision_recall/add_3Add9dnn/head/metrics/auc_precision_recall/true_positives/read-dnn/head/metrics/auc_precision_recall/add_3/y*
_output_shapes	
:�*
T0
�
+dnn/head/metrics/auc_precision_recall/add_4Add9dnn/head/metrics/auc_precision_recall/true_positives/read:dnn/head/metrics/auc_precision_recall/false_positives/read*
T0*
_output_shapes	
:�
r
-dnn/head/metrics/auc_precision_recall/add_5/yConst*
valueB
 *�7�5*
dtype0*
_output_shapes
: 
�
+dnn/head/metrics/auc_precision_recall/add_5Add+dnn/head/metrics/auc_precision_recall/add_4-dnn/head/metrics/auc_precision_recall/add_5/y*
_output_shapes	
:�*
T0
�
+dnn/head/metrics/auc_precision_recall/div_1RealDiv+dnn/head/metrics/auc_precision_recall/add_3+dnn/head/metrics/auc_precision_recall/add_5*
_output_shapes	
:�*
T0
�
;dnn/head/metrics/auc_precision_recall/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
=dnn/head/metrics/auc_precision_recall/strided_slice_1/stack_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
=dnn/head/metrics/auc_precision_recall/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
5dnn/head/metrics/auc_precision_recall/strided_slice_1StridedSlice)dnn/head/metrics/auc_precision_recall/div;dnn/head/metrics/auc_precision_recall/strided_slice_1/stack=dnn/head/metrics/auc_precision_recall/strided_slice_1/stack_1=dnn/head/metrics/auc_precision_recall/strided_slice_1/stack_2*
end_mask *
_output_shapes	
:�*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask 
�
;dnn/head/metrics/auc_precision_recall/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
�
=dnn/head/metrics/auc_precision_recall/strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
�
=dnn/head/metrics/auc_precision_recall/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
5dnn/head/metrics/auc_precision_recall/strided_slice_2StridedSlice)dnn/head/metrics/auc_precision_recall/div;dnn/head/metrics/auc_precision_recall/strided_slice_2/stack=dnn/head/metrics/auc_precision_recall/strided_slice_2/stack_1=dnn/head/metrics/auc_precision_recall/strided_slice_2/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes	
:�*
Index0*
T0
�
)dnn/head/metrics/auc_precision_recall/subSub5dnn/head/metrics/auc_precision_recall/strided_slice_15dnn/head/metrics/auc_precision_recall/strided_slice_2*
T0*
_output_shapes	
:�
�
;dnn/head/metrics/auc_precision_recall/strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
=dnn/head/metrics/auc_precision_recall/strided_slice_3/stack_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
=dnn/head/metrics/auc_precision_recall/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
5dnn/head/metrics/auc_precision_recall/strided_slice_3StridedSlice+dnn/head/metrics/auc_precision_recall/div_1;dnn/head/metrics/auc_precision_recall/strided_slice_3/stack=dnn/head/metrics/auc_precision_recall/strided_slice_3/stack_1=dnn/head/metrics/auc_precision_recall/strided_slice_3/stack_2*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask *
_output_shapes	
:�
�
;dnn/head/metrics/auc_precision_recall/strided_slice_4/stackConst*
valueB:*
dtype0*
_output_shapes
:
�
=dnn/head/metrics/auc_precision_recall/strided_slice_4/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
=dnn/head/metrics/auc_precision_recall/strided_slice_4/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
5dnn/head/metrics/auc_precision_recall/strided_slice_4StridedSlice+dnn/head/metrics/auc_precision_recall/div_1;dnn/head/metrics/auc_precision_recall/strided_slice_4/stack=dnn/head/metrics/auc_precision_recall/strided_slice_4/stack_1=dnn/head/metrics/auc_precision_recall/strided_slice_4/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask*
_output_shapes	
:�*
Index0*
T0
�
+dnn/head/metrics/auc_precision_recall/add_6Add5dnn/head/metrics/auc_precision_recall/strided_slice_35dnn/head/metrics/auc_precision_recall/strided_slice_4*
T0*
_output_shapes	
:�
t
/dnn/head/metrics/auc_precision_recall/truediv/yConst*
dtype0*
_output_shapes
: *
valueB
 *   @
�
-dnn/head/metrics/auc_precision_recall/truedivRealDiv+dnn/head/metrics/auc_precision_recall/add_6/dnn/head/metrics/auc_precision_recall/truediv/y*
T0*
_output_shapes	
:�
�
)dnn/head/metrics/auc_precision_recall/MulMul)dnn/head/metrics/auc_precision_recall/sub-dnn/head/metrics/auc_precision_recall/truediv*
_output_shapes	
:�*
T0
w
-dnn/head/metrics/auc_precision_recall/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
+dnn/head/metrics/auc_precision_recall/valueSum)dnn/head/metrics/auc_precision_recall/Mul-dnn/head/metrics/auc_precision_recall/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
r
-dnn/head/metrics/auc_precision_recall/add_7/yConst*
_output_shapes
: *
valueB
 *�7�5*
dtype0
�
+dnn/head/metrics/auc_precision_recall/add_7Add/dnn/head/metrics/auc_precision_recall/AssignAdd-dnn/head/metrics/auc_precision_recall/add_7/y*
T0*
_output_shapes	
:�
�
+dnn/head/metrics/auc_precision_recall/add_8Add/dnn/head/metrics/auc_precision_recall/AssignAdd1dnn/head/metrics/auc_precision_recall/AssignAdd_1*
T0*
_output_shapes	
:�
r
-dnn/head/metrics/auc_precision_recall/add_9/yConst*
valueB
 *�7�5*
dtype0*
_output_shapes
: 
�
+dnn/head/metrics/auc_precision_recall/add_9Add+dnn/head/metrics/auc_precision_recall/add_8-dnn/head/metrics/auc_precision_recall/add_9/y*
_output_shapes	
:�*
T0
�
+dnn/head/metrics/auc_precision_recall/div_2RealDiv+dnn/head/metrics/auc_precision_recall/add_7+dnn/head/metrics/auc_precision_recall/add_9*
_output_shapes	
:�*
T0
s
.dnn/head/metrics/auc_precision_recall/add_10/yConst*
valueB
 *�7�5*
dtype0*
_output_shapes
: 
�
,dnn/head/metrics/auc_precision_recall/add_10Add/dnn/head/metrics/auc_precision_recall/AssignAdd.dnn/head/metrics/auc_precision_recall/add_10/y*
_output_shapes	
:�*
T0
�
,dnn/head/metrics/auc_precision_recall/add_11Add/dnn/head/metrics/auc_precision_recall/AssignAdd1dnn/head/metrics/auc_precision_recall/AssignAdd_3*
_output_shapes	
:�*
T0
s
.dnn/head/metrics/auc_precision_recall/add_12/yConst*
valueB
 *�7�5*
dtype0*
_output_shapes
: 
�
,dnn/head/metrics/auc_precision_recall/add_12Add,dnn/head/metrics/auc_precision_recall/add_11.dnn/head/metrics/auc_precision_recall/add_12/y*
T0*
_output_shapes	
:�
�
+dnn/head/metrics/auc_precision_recall/div_3RealDiv,dnn/head/metrics/auc_precision_recall/add_10,dnn/head/metrics/auc_precision_recall/add_12*
T0*
_output_shapes	
:�
�
;dnn/head/metrics/auc_precision_recall/strided_slice_5/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
=dnn/head/metrics/auc_precision_recall/strided_slice_5/stack_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
=dnn/head/metrics/auc_precision_recall/strided_slice_5/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
5dnn/head/metrics/auc_precision_recall/strided_slice_5StridedSlice+dnn/head/metrics/auc_precision_recall/div_2;dnn/head/metrics/auc_precision_recall/strided_slice_5/stack=dnn/head/metrics/auc_precision_recall/strided_slice_5/stack_1=dnn/head/metrics/auc_precision_recall/strided_slice_5/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes	
:�
�
;dnn/head/metrics/auc_precision_recall/strided_slice_6/stackConst*
valueB:*
dtype0*
_output_shapes
:
�
=dnn/head/metrics/auc_precision_recall/strided_slice_6/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
�
=dnn/head/metrics/auc_precision_recall/strided_slice_6/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
5dnn/head/metrics/auc_precision_recall/strided_slice_6StridedSlice+dnn/head/metrics/auc_precision_recall/div_2;dnn/head/metrics/auc_precision_recall/strided_slice_6/stack=dnn/head/metrics/auc_precision_recall/strided_slice_6/stack_1=dnn/head/metrics/auc_precision_recall/strided_slice_6/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes	
:�*
T0*
Index0*
shrink_axis_mask 
�
+dnn/head/metrics/auc_precision_recall/sub_1Sub5dnn/head/metrics/auc_precision_recall/strided_slice_55dnn/head/metrics/auc_precision_recall/strided_slice_6*
T0*
_output_shapes	
:�
�
;dnn/head/metrics/auc_precision_recall/strided_slice_7/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
=dnn/head/metrics/auc_precision_recall/strided_slice_7/stack_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
=dnn/head/metrics/auc_precision_recall/strided_slice_7/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
5dnn/head/metrics/auc_precision_recall/strided_slice_7StridedSlice+dnn/head/metrics/auc_precision_recall/div_3;dnn/head/metrics/auc_precision_recall/strided_slice_7/stack=dnn/head/metrics/auc_precision_recall/strided_slice_7/stack_1=dnn/head/metrics/auc_precision_recall/strided_slice_7/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes	
:�*
Index0*
T0*
shrink_axis_mask 
�
;dnn/head/metrics/auc_precision_recall/strided_slice_8/stackConst*
valueB:*
dtype0*
_output_shapes
:
�
=dnn/head/metrics/auc_precision_recall/strided_slice_8/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
=dnn/head/metrics/auc_precision_recall/strided_slice_8/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
�
5dnn/head/metrics/auc_precision_recall/strided_slice_8StridedSlice+dnn/head/metrics/auc_precision_recall/div_3;dnn/head/metrics/auc_precision_recall/strided_slice_8/stack=dnn/head/metrics/auc_precision_recall/strided_slice_8/stack_1=dnn/head/metrics/auc_precision_recall/strided_slice_8/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes	
:�*
Index0*
T0*
shrink_axis_mask 
�
,dnn/head/metrics/auc_precision_recall/add_13Add5dnn/head/metrics/auc_precision_recall/strided_slice_75dnn/head/metrics/auc_precision_recall/strided_slice_8*
T0*
_output_shapes	
:�
v
1dnn/head/metrics/auc_precision_recall/truediv_1/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
/dnn/head/metrics/auc_precision_recall/truediv_1RealDiv,dnn/head/metrics/auc_precision_recall/add_131dnn/head/metrics/auc_precision_recall/truediv_1/y*
T0*
_output_shapes	
:�
�
+dnn/head/metrics/auc_precision_recall/Mul_1Mul+dnn/head/metrics/auc_precision_recall/sub_1/dnn/head/metrics/auc_precision_recall/truediv_1*
_output_shapes	
:�*
T0
w
-dnn/head/metrics/auc_precision_recall/Const_2Const*
valueB: *
dtype0*
_output_shapes
:
�
/dnn/head/metrics/auc_precision_recall/update_opSum+dnn/head/metrics/auc_precision_recall/Mul_1-dnn/head/metrics/auc_precision_recall/Const_2*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
mean/total/Initializer/zerosConst*
dtype0*
_output_shapes
: *
_class
loc:@mean/total*
valueB
 *    
�

mean/total
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@mean/total*
	container *
shape: 
�
mean/total/AssignAssign
mean/totalmean/total/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@mean/total
g
mean/total/readIdentity
mean/total*
_output_shapes
: *
T0*
_class
loc:@mean/total
�
mean/count/Initializer/zerosConst*
_class
loc:@mean/count*
valueB
 *    *
dtype0*
_output_shapes
: 
�

mean/count
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@mean/count*
	container 
�
mean/count/AssignAssign
mean/countmean/count/Initializer/zeros*
_class
loc:@mean/count*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
g
mean/count/readIdentity
mean/count*
_output_shapes
: *
T0*
_class
loc:@mean/count
K
	mean/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
Q
mean/ToFloat_1Cast	mean/Size*
_output_shapes
: *

DstT0*

SrcT0
M

mean/ConstConst*
valueB *
dtype0*
_output_shapes
: 
u
mean/SumSumdnn/head/weighted_loss/Sum
mean/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
mean/AssignAdd	AssignAdd
mean/totalmean/Sum*
_output_shapes
: *
use_locking( *
T0*
_class
loc:@mean/total
�
mean/AssignAdd_1	AssignAdd
mean/countmean/ToFloat_1^dnn/head/weighted_loss/Sum*
use_locking( *
T0*
_class
loc:@mean/count*
_output_shapes
: 
Z
mean/truedivRealDivmean/total/readmean/count/read*
T0*
_output_shapes
: 
T
mean/zeros_likeConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
mean/GreaterGreatermean/count/readmean/zeros_like*
_output_shapes
: *
T0
b

mean/valueSelectmean/Greatermean/truedivmean/zeros_like*
T0*
_output_shapes
: 
\
mean/truediv_1RealDivmean/AssignAddmean/AssignAdd_1*
T0*
_output_shapes
: 
V
mean/zeros_like_1Const*
_output_shapes
: *
valueB
 *    *
dtype0
_
mean/Greater_1Greatermean/AssignAdd_1mean/zeros_like_1*
T0*
_output_shapes
: 
l
mean/update_opSelectmean/Greater_1mean/truediv_1mean/zeros_like_1*
_output_shapes
: *
T0
�

group_depsNoOp$^dnn/head/metrics/accuracy/update_op-^dnn/head/metrics/accuracy_baseline/update_op^dnn/head/metrics/auc/update_op0^dnn/head/metrics/auc_precision_recall/update_op(^dnn/head/metrics/average_loss/update_op&^dnn/head/metrics/label/mean/update_op%^dnn/head/metrics/precision/update_op+^dnn/head/metrics/prediction/mean/update_op"^dnn/head/metrics/recall/update_op^mean/update_op
{
eval_step/Initializer/zerosConst*
_class
loc:@eval_step*
value	B	 R *
dtype0	*
_output_shapes
: 
�
	eval_step
VariableV2*
	container *
shape: *
dtype0	*
_output_shapes
: *
shared_name *
_class
loc:@eval_step
�
eval_step/AssignAssign	eval_stepeval_step/Initializer/zeros*
use_locking(*
T0	*
_class
loc:@eval_step*
validate_shape(*
_output_shapes
: 
d
eval_step/readIdentity	eval_step*
T0	*
_class
loc:@eval_step*
_output_shapes
: 
Q
AssignAdd/valueConst*
value	B	 R*
dtype0	*
_output_shapes
: 
�
	AssignAdd	AssignAdd	eval_stepAssignAdd/value*
_output_shapes
: *
use_locking(*
T0	*
_class
loc:@eval_step
U
readIdentity	eval_step
^AssignAdd^group_deps*
_output_shapes
: *
T0	
;
IdentityIdentityread*
_output_shapes
: *
T0	
�
initNoOp%^dnn/hiddenlayer_0/bias/part_0/Assign'^dnn/hiddenlayer_0/kernel/part_0/Assign%^dnn/hiddenlayer_1/bias/part_0/Assign'^dnn/hiddenlayer_1/kernel/part_0/Assign^dnn/logits/bias/part_0/Assign ^dnn/logits/kernel/part_0/Assign^global_step/Assign

init_1NoOp
$
group_deps_1NoOp^init^init_1
�
4report_uninitialized_variables/IsVariableInitializedIsVariableInitializedglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_1IsVariableInitializeddnn/hiddenlayer_0/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_2IsVariableInitializeddnn/hiddenlayer_0/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_3IsVariableInitializeddnn/hiddenlayer_1/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_4IsVariableInitializeddnn/hiddenlayer_1/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_5IsVariableInitializeddnn/logits/kernel/part_0*
_output_shapes
: *+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0
�
6report_uninitialized_variables/IsVariableInitialized_6IsVariableInitializeddnn/logits/bias/part_0*)
_class
loc:@dnn/logits/bias/part_0*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_7IsVariableInitialized!dnn/head/metrics/label/mean/total*4
_class*
(&loc:@dnn/head/metrics/label/mean/total*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_8IsVariableInitialized!dnn/head/metrics/label/mean/count*
_output_shapes
: *4
_class*
(&loc:@dnn/head/metrics/label/mean/count*
dtype0
�
6report_uninitialized_variables/IsVariableInitialized_9IsVariableInitialized#dnn/head/metrics/average_loss/total*6
_class,
*(loc:@dnn/head/metrics/average_loss/total*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_10IsVariableInitialized#dnn/head/metrics/average_loss/count*6
_class,
*(loc:@dnn/head/metrics/average_loss/count*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_11IsVariableInitializeddnn/head/metrics/accuracy/total*2
_class(
&$loc:@dnn/head/metrics/accuracy/total*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_12IsVariableInitializeddnn/head/metrics/accuracy/count*2
_class(
&$loc:@dnn/head/metrics/accuracy/count*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_13IsVariableInitialized/dnn/head/metrics/precision/true_positives/count*B
_class8
64loc:@dnn/head/metrics/precision/true_positives/count*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_14IsVariableInitialized0dnn/head/metrics/precision/false_positives/count*
_output_shapes
: *C
_class9
75loc:@dnn/head/metrics/precision/false_positives/count*
dtype0
�
7report_uninitialized_variables/IsVariableInitialized_15IsVariableInitialized,dnn/head/metrics/recall/true_positives/count*?
_class5
31loc:@dnn/head/metrics/recall/true_positives/count*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_16IsVariableInitialized-dnn/head/metrics/recall/false_negatives/count*@
_class6
42loc:@dnn/head/metrics/recall/false_negatives/count*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_17IsVariableInitialized&dnn/head/metrics/prediction/mean/total*
_output_shapes
: *9
_class/
-+loc:@dnn/head/metrics/prediction/mean/total*
dtype0
�
7report_uninitialized_variables/IsVariableInitialized_18IsVariableInitialized&dnn/head/metrics/prediction/mean/count*9
_class/
-+loc:@dnn/head/metrics/prediction/mean/count*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_19IsVariableInitialized#dnn/head/metrics/auc/true_positives*
_output_shapes
: *6
_class,
*(loc:@dnn/head/metrics/auc/true_positives*
dtype0
�
7report_uninitialized_variables/IsVariableInitialized_20IsVariableInitialized$dnn/head/metrics/auc/false_negatives*7
_class-
+)loc:@dnn/head/metrics/auc/false_negatives*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_21IsVariableInitialized#dnn/head/metrics/auc/true_negatives*
dtype0*
_output_shapes
: *6
_class,
*(loc:@dnn/head/metrics/auc/true_negatives
�
7report_uninitialized_variables/IsVariableInitialized_22IsVariableInitialized$dnn/head/metrics/auc/false_positives*7
_class-
+)loc:@dnn/head/metrics/auc/false_positives*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_23IsVariableInitialized4dnn/head/metrics/auc_precision_recall/true_positives*G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_positives*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_24IsVariableInitialized5dnn/head/metrics/auc_precision_recall/false_negatives*
dtype0*
_output_shapes
: *H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_negatives
�
7report_uninitialized_variables/IsVariableInitialized_25IsVariableInitialized4dnn/head/metrics/auc_precision_recall/true_negatives*G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_negatives*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_26IsVariableInitialized5dnn/head/metrics/auc_precision_recall/false_positives*H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_positives*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_27IsVariableInitialized
mean/total*
dtype0*
_output_shapes
: *
_class
loc:@mean/total
�
7report_uninitialized_variables/IsVariableInitialized_28IsVariableInitialized
mean/count*
_class
loc:@mean/count*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_29IsVariableInitialized	eval_step*
_class
loc:@eval_step*
dtype0	*
_output_shapes
: 
�
$report_uninitialized_variables/stackPack4report_uninitialized_variables/IsVariableInitialized6report_uninitialized_variables/IsVariableInitialized_16report_uninitialized_variables/IsVariableInitialized_26report_uninitialized_variables/IsVariableInitialized_36report_uninitialized_variables/IsVariableInitialized_46report_uninitialized_variables/IsVariableInitialized_56report_uninitialized_variables/IsVariableInitialized_66report_uninitialized_variables/IsVariableInitialized_76report_uninitialized_variables/IsVariableInitialized_86report_uninitialized_variables/IsVariableInitialized_97report_uninitialized_variables/IsVariableInitialized_107report_uninitialized_variables/IsVariableInitialized_117report_uninitialized_variables/IsVariableInitialized_127report_uninitialized_variables/IsVariableInitialized_137report_uninitialized_variables/IsVariableInitialized_147report_uninitialized_variables/IsVariableInitialized_157report_uninitialized_variables/IsVariableInitialized_167report_uninitialized_variables/IsVariableInitialized_177report_uninitialized_variables/IsVariableInitialized_187report_uninitialized_variables/IsVariableInitialized_197report_uninitialized_variables/IsVariableInitialized_207report_uninitialized_variables/IsVariableInitialized_217report_uninitialized_variables/IsVariableInitialized_227report_uninitialized_variables/IsVariableInitialized_237report_uninitialized_variables/IsVariableInitialized_247report_uninitialized_variables/IsVariableInitialized_257report_uninitialized_variables/IsVariableInitialized_267report_uninitialized_variables/IsVariableInitialized_277report_uninitialized_variables/IsVariableInitialized_287report_uninitialized_variables/IsVariableInitialized_29"/device:CPU:0*
T0
*

axis *
N*
_output_shapes
:
�
)report_uninitialized_variables/LogicalNot
LogicalNot$report_uninitialized_variables/stack"/device:CPU:0*
_output_shapes
:
�	
$report_uninitialized_variables/ConstConst"/device:CPU:0*�
value�B�Bglobal_stepBdnn/hiddenlayer_0/kernel/part_0Bdnn/hiddenlayer_0/bias/part_0Bdnn/hiddenlayer_1/kernel/part_0Bdnn/hiddenlayer_1/bias/part_0Bdnn/logits/kernel/part_0Bdnn/logits/bias/part_0B!dnn/head/metrics/label/mean/totalB!dnn/head/metrics/label/mean/countB#dnn/head/metrics/average_loss/totalB#dnn/head/metrics/average_loss/countBdnn/head/metrics/accuracy/totalBdnn/head/metrics/accuracy/countB/dnn/head/metrics/precision/true_positives/countB0dnn/head/metrics/precision/false_positives/countB,dnn/head/metrics/recall/true_positives/countB-dnn/head/metrics/recall/false_negatives/countB&dnn/head/metrics/prediction/mean/totalB&dnn/head/metrics/prediction/mean/countB#dnn/head/metrics/auc/true_positivesB$dnn/head/metrics/auc/false_negativesB#dnn/head/metrics/auc/true_negativesB$dnn/head/metrics/auc/false_positivesB4dnn/head/metrics/auc_precision_recall/true_positivesB5dnn/head/metrics/auc_precision_recall/false_negativesB4dnn/head/metrics/auc_precision_recall/true_negativesB5dnn/head/metrics/auc_precision_recall/false_positivesB
mean/totalB
mean/countB	eval_step*
dtype0*
_output_shapes
:
�
1report_uninitialized_variables/boolean_mask/ShapeConst"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
?report_uninitialized_variables/boolean_mask/strided_slice/stackConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB: 
�
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_1Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2Const"/device:CPU:0*
_output_shapes
:*
valueB:*
dtype0
�
9report_uninitialized_variables/boolean_mask/strided_sliceStridedSlice1report_uninitialized_variables/boolean_mask/Shape?report_uninitialized_variables/boolean_mask/strided_slice/stackAreport_uninitialized_variables/boolean_mask/strided_slice/stack_1Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2"/device:CPU:0*
new_axis_mask *
end_mask *
_output_shapes
:*
Index0*
T0*
shrink_axis_mask *

begin_mask *
ellipsis_mask 
�
Breport_uninitialized_variables/boolean_mask/Prod/reduction_indicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB: 
�
0report_uninitialized_variables/boolean_mask/ProdProd9report_uninitialized_variables/boolean_mask/strided_sliceBreport_uninitialized_variables/boolean_mask/Prod/reduction_indices"/device:CPU:0*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
3report_uninitialized_variables/boolean_mask/Shape_1Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Const"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
;report_uninitialized_variables/boolean_mask/strided_slice_1StridedSlice3report_uninitialized_variables/boolean_mask/Shape_1Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackCreport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2"/device:CPU:0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
�
3report_uninitialized_variables/boolean_mask/Shape_2Const"/device:CPU:0*
_output_shapes
:*
valueB:*
dtype0
�
Areport_uninitialized_variables/boolean_mask/strided_slice_2/stackConst"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
Creport_uninitialized_variables/boolean_mask/strided_slice_2/stack_1Const"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
Creport_uninitialized_variables/boolean_mask/strided_slice_2/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
;report_uninitialized_variables/boolean_mask/strided_slice_2StridedSlice3report_uninitialized_variables/boolean_mask/Shape_2Areport_uninitialized_variables/boolean_mask/strided_slice_2/stackCreport_uninitialized_variables/boolean_mask/strided_slice_2/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_2/stack_2"/device:CPU:0*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
: *
T0*
Index0*
shrink_axis_mask 
�
;report_uninitialized_variables/boolean_mask/concat/values_1Pack0report_uninitialized_variables/boolean_mask/Prod"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:
�
7report_uninitialized_variables/boolean_mask/concat/axisConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : 
�
2report_uninitialized_variables/boolean_mask/concatConcatV2;report_uninitialized_variables/boolean_mask/strided_slice_1;report_uninitialized_variables/boolean_mask/concat/values_1;report_uninitialized_variables/boolean_mask/strided_slice_27report_uninitialized_variables/boolean_mask/concat/axis"/device:CPU:0*
T0*
N*
_output_shapes
:*

Tidx0
�
3report_uninitialized_variables/boolean_mask/ReshapeReshape$report_uninitialized_variables/Const2report_uninitialized_variables/boolean_mask/concat"/device:CPU:0*
T0*
Tshape0*
_output_shapes
:
�
;report_uninitialized_variables/boolean_mask/Reshape_1/shapeConst"/device:CPU:0*
valueB:
���������*
dtype0*
_output_shapes
:
�
5report_uninitialized_variables/boolean_mask/Reshape_1Reshape)report_uninitialized_variables/LogicalNot;report_uninitialized_variables/boolean_mask/Reshape_1/shape"/device:CPU:0*
T0
*
Tshape0*
_output_shapes
:
�
1report_uninitialized_variables/boolean_mask/WhereWhere5report_uninitialized_variables/boolean_mask/Reshape_1"/device:CPU:0*
T0
*'
_output_shapes
:���������
�
3report_uninitialized_variables/boolean_mask/SqueezeSqueeze1report_uninitialized_variables/boolean_mask/Where"/device:CPU:0*
T0	*#
_output_shapes
:���������*
squeeze_dims

�
9report_uninitialized_variables/boolean_mask/GatherV2/axisConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : 
�
4report_uninitialized_variables/boolean_mask/GatherV2GatherV23report_uninitialized_variables/boolean_mask/Reshape3report_uninitialized_variables/boolean_mask/Squeeze9report_uninitialized_variables/boolean_mask/GatherV2/axis"/device:CPU:0*
Tindices0	*
Tparams0*#
_output_shapes
:���������*
Taxis0
v
$report_uninitialized_resources/ConstConst"/device:CPU:0*
dtype0*
_output_shapes
: *
valueB 
M
concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
concatConcatV24report_uninitialized_variables/boolean_mask/GatherV2$report_uninitialized_resources/Constconcat/axis*

Tidx0*
T0*
N*#
_output_shapes
:���������
�
6report_uninitialized_variables_1/IsVariableInitializedIsVariableInitializedglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_1IsVariableInitializeddnn/hiddenlayer_0/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_2IsVariableInitializeddnn/hiddenlayer_0/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_3IsVariableInitializeddnn/hiddenlayer_1/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_4IsVariableInitializeddnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes
: *0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0
�
8report_uninitialized_variables_1/IsVariableInitialized_5IsVariableInitializeddnn/logits/kernel/part_0*
dtype0*
_output_shapes
: *+
_class!
loc:@dnn/logits/kernel/part_0
�
8report_uninitialized_variables_1/IsVariableInitialized_6IsVariableInitializeddnn/logits/bias/part_0*)
_class
loc:@dnn/logits/bias/part_0*
dtype0*
_output_shapes
: 
�
&report_uninitialized_variables_1/stackPack6report_uninitialized_variables_1/IsVariableInitialized8report_uninitialized_variables_1/IsVariableInitialized_18report_uninitialized_variables_1/IsVariableInitialized_28report_uninitialized_variables_1/IsVariableInitialized_38report_uninitialized_variables_1/IsVariableInitialized_48report_uninitialized_variables_1/IsVariableInitialized_58report_uninitialized_variables_1/IsVariableInitialized_6"/device:CPU:0*
T0
*

axis *
N*
_output_shapes
:
�
+report_uninitialized_variables_1/LogicalNot
LogicalNot&report_uninitialized_variables_1/stack"/device:CPU:0*
_output_shapes
:
�
&report_uninitialized_variables_1/ConstConst"/device:CPU:0*�
value�B�Bglobal_stepBdnn/hiddenlayer_0/kernel/part_0Bdnn/hiddenlayer_0/bias/part_0Bdnn/hiddenlayer_1/kernel/part_0Bdnn/hiddenlayer_1/bias/part_0Bdnn/logits/kernel/part_0Bdnn/logits/bias/part_0*
dtype0*
_output_shapes
:
�
3report_uninitialized_variables_1/boolean_mask/ShapeConst"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
Areport_uninitialized_variables_1/boolean_mask/strided_slice/stackConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB: 
�
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Const"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:
�
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
;report_uninitialized_variables_1/boolean_mask/strided_sliceStridedSlice3report_uninitialized_variables_1/boolean_mask/ShapeAreport_uninitialized_variables_1/boolean_mask/strided_slice/stackCreport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2"/device:CPU:0*
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:
�
Dreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indicesConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
2report_uninitialized_variables_1/boolean_mask/ProdProd;report_uninitialized_variables_1/boolean_mask/strided_sliceDreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indices"/device:CPU:0*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
5report_uninitialized_variables_1/boolean_mask/Shape_1Const"/device:CPU:0*
_output_shapes
:*
valueB:*
dtype0
�
Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB: 
�
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Const"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
=report_uninitialized_variables_1/boolean_mask/strided_slice_1StridedSlice5report_uninitialized_variables_1/boolean_mask/Shape_1Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackEreport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2"/device:CPU:0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
�
5report_uninitialized_variables_1/boolean_mask/Shape_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
Creport_uninitialized_variables_1/boolean_mask/strided_slice_2/stackConst"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_1Const"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_2Const"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:
�
=report_uninitialized_variables_1/boolean_mask/strided_slice_2StridedSlice5report_uninitialized_variables_1/boolean_mask/Shape_2Creport_uninitialized_variables_1/boolean_mask/strided_slice_2/stackEreport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_1Ereport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_2"/device:CPU:0*
_output_shapes
: *
Index0*
T0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask
�
=report_uninitialized_variables_1/boolean_mask/concat/values_1Pack2report_uninitialized_variables_1/boolean_mask/Prod"/device:CPU:0*
_output_shapes
:*
T0*

axis *
N
�
9report_uninitialized_variables_1/boolean_mask/concat/axisConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
4report_uninitialized_variables_1/boolean_mask/concatConcatV2=report_uninitialized_variables_1/boolean_mask/strided_slice_1=report_uninitialized_variables_1/boolean_mask/concat/values_1=report_uninitialized_variables_1/boolean_mask/strided_slice_29report_uninitialized_variables_1/boolean_mask/concat/axis"/device:CPU:0*
T0*
N*
_output_shapes
:*

Tidx0
�
5report_uninitialized_variables_1/boolean_mask/ReshapeReshape&report_uninitialized_variables_1/Const4report_uninitialized_variables_1/boolean_mask/concat"/device:CPU:0*
_output_shapes
:*
T0*
Tshape0
�
=report_uninitialized_variables_1/boolean_mask/Reshape_1/shapeConst"/device:CPU:0*
valueB:
���������*
dtype0*
_output_shapes
:
�
7report_uninitialized_variables_1/boolean_mask/Reshape_1Reshape+report_uninitialized_variables_1/LogicalNot=report_uninitialized_variables_1/boolean_mask/Reshape_1/shape"/device:CPU:0*
_output_shapes
:*
T0
*
Tshape0
�
3report_uninitialized_variables_1/boolean_mask/WhereWhere7report_uninitialized_variables_1/boolean_mask/Reshape_1"/device:CPU:0*
T0
*'
_output_shapes
:���������
�
5report_uninitialized_variables_1/boolean_mask/SqueezeSqueeze3report_uninitialized_variables_1/boolean_mask/Where"/device:CPU:0*#
_output_shapes
:���������*
squeeze_dims
*
T0	
�
;report_uninitialized_variables_1/boolean_mask/GatherV2/axisConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables_1/boolean_mask/GatherV2GatherV25report_uninitialized_variables_1/boolean_mask/Reshape5report_uninitialized_variables_1/boolean_mask/Squeeze;report_uninitialized_variables_1/boolean_mask/GatherV2/axis"/device:CPU:0*
Tindices0	*
Tparams0*#
_output_shapes
:���������*
Taxis0
�
init_2NoOp'^dnn/head/metrics/accuracy/count/Assign'^dnn/head/metrics/accuracy/total/Assign,^dnn/head/metrics/auc/false_negatives/Assign,^dnn/head/metrics/auc/false_positives/Assign+^dnn/head/metrics/auc/true_negatives/Assign+^dnn/head/metrics/auc/true_positives/Assign=^dnn/head/metrics/auc_precision_recall/false_negatives/Assign=^dnn/head/metrics/auc_precision_recall/false_positives/Assign<^dnn/head/metrics/auc_precision_recall/true_negatives/Assign<^dnn/head/metrics/auc_precision_recall/true_positives/Assign+^dnn/head/metrics/average_loss/count/Assign+^dnn/head/metrics/average_loss/total/Assign)^dnn/head/metrics/label/mean/count/Assign)^dnn/head/metrics/label/mean/total/Assign8^dnn/head/metrics/precision/false_positives/count/Assign7^dnn/head/metrics/precision/true_positives/count/Assign.^dnn/head/metrics/prediction/mean/count/Assign.^dnn/head/metrics/prediction/mean/total/Assign5^dnn/head/metrics/recall/false_negatives/count/Assign4^dnn/head/metrics/recall/true_positives/count/Assign^eval_step/Assign^mean/count/Assign^mean/total/Assign

init_all_tablesNoOp

init_3NoOp
8
group_deps_2NoOp^init_2^init_3^init_all_tables
�
Merge/MergeSummaryMergeSummaryHenqueue_input/queue/enqueue_input/fifo_queuefraction_over_0_of_1000_full-dnn/dnn/hiddenlayer_0/fraction_of_zero_values dnn/dnn/hiddenlayer_0/activation-dnn/dnn/hiddenlayer_1/fraction_of_zero_values dnn/dnn/hiddenlayer_1/activation&dnn/dnn/logits/fraction_of_zero_valuesdnn/dnn/logits/activation*
N*
_output_shapes
: 
P

save/ConstConst*
dtype0*
_output_shapes
: *
valueB Bmodel
�
save/StringJoin/inputs_1Const*<
value3B1 B+_temp_2643c5b94d414b26b776ae06de9d9710/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : 
�
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst"/device:CPU:0*�
value�B�Bdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/logits/biasBdnn/logits/kernelBglobal_step*
dtype0*
_output_shapes
:
�
save/SaveV2/shape_and_slicesConst"/device:CPU:0*i
value`B^B	256 0,256B4096 256 0,4096:0,256B32 0,32B256 32 0,256:0,32B1 0,1B32 1 0,32:0,1B *
dtype0*
_output_shapes
:
�
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices"dnn/hiddenlayer_0/bias/part_0/read$dnn/hiddenlayer_0/kernel/part_0/read"dnn/hiddenlayer_1/bias/part_0/read$dnn/hiddenlayer_1/kernel/part_0/readdnn/logits/bias/part_0/readdnn/logits/kernel/part_0/readglobal_step"/device:CPU:0*
dtypes
	2	
�
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
�
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:
�
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(
�
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
_output_shapes
: *
T0
�
save/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�Bdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/logits/biasBdnn/logits/kernelBglobal_step*
dtype0*
_output_shapes
:
�
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*i
value`B^B	256 0,256B4096 256 0,4096:0,256B32 0,32B256 32 0,256:0,32B1 0,1B32 1 0,32:0,1B *
dtype0*
_output_shapes
:
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*L
_output_shapes:
8:�:
� �: :	� :: :*
dtypes
	2	
�
save/AssignAssigndnn/hiddenlayer_0/bias/part_0save/RestoreV2*
use_locking(*
T0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
validate_shape(*
_output_shapes	
:�
�
save/Assign_1Assigndnn/hiddenlayer_0/kernel/part_0save/RestoreV2:1*
use_locking(*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
validate_shape(* 
_output_shapes
:
� �
�
save/Assign_2Assigndnn/hiddenlayer_1/bias/part_0save/RestoreV2:2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0
�
save/Assign_3Assigndnn/hiddenlayer_1/kernel/part_0save/RestoreV2:3*
use_locking(*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
validate_shape(*
_output_shapes
:	� 
�
save/Assign_4Assigndnn/logits/bias/part_0save/RestoreV2:4*)
_class
loc:@dnn/logits/bias/part_0*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
save/Assign_5Assigndnn/logits/kernel/part_0save/RestoreV2:5*
validate_shape(*
_output_shapes

: *
use_locking(*
T0*+
_class!
loc:@dnn/logits/kernel/part_0
�
save/Assign_6Assignglobal_stepsave/RestoreV2:6*
use_locking(*
T0	*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: 
�
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6
-
save/restore_allNoOp^save/restore_shard"����     A��	A�/:��AJ��
�1�0
:
Add
x"T
y"T
z"T"
Ttype:
2	
h
All	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	
�
ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
�
AsString

input"T

output"
Ttype:
	2	
"
	precisionint���������"

scientificbool( "
shortestbool( "
widthint���������"
fillstring 
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeint�
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
s
	AssignAdd
ref"T�

value"T

output_ref"T�" 
Ttype:
2	"
use_lockingbool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
DenseToDenseSetOperation	
set1"T	
set2"T
result_indices	
result_values"T
result_shape	"
set_operationstring"
validate_indicesbool("
Ttype:
	2	
B
Equal
x"T
y"T
z
"
Ttype:
2	
�
,
Exp
x"T
y"T"
Ttype:

2
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
�
FIFOQueueV2

handle"!
component_types
list(type)(0"
shapeslist(shape)
 ("
capacityint���������"
	containerstring "
shared_namestring �
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
V
HistogramSummary
tag
values"T
summary"
Ttype0:
2	
.
Identity

input"T
output"T"	
Ttype
N
IsVariableInitialized
ref"dtype�
is_initialized
"
dtypetype�
:
Less
x"T
y"T
z
"
Ttype:
2	
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
.
Log1p
x"T
y"T"
Ttype:

2
$

LogicalAnd
x

y

z
�


LogicalNot
x

y

p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
;
Maximum
x"T
y"T
z"T"
Ttype:

2	�
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
8
MergeSummary
inputs*N
summary"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
B
QueueCloseV2

handle"#
cancel_pending_enqueuesbool( �
�
QueueDequeueUpToV2

handle
n

components2component_types"!
component_types
list(type)(0"

timeout_msint����������
}
QueueEnqueueManyV2

handle

components2Tcomponents"
Tcomponents
list(type)(0"

timeout_msint����������
&
QueueSizeV2

handle
size�
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
)
Rank

input"T

output"	
Ttype
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
D
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �
E
Where

input"T	
index	"%
Ttype0
:
2	

&
	ZerosLike
x"T
y"T"	
Ttype*1.8.02v1.8.0-0-g93bc2e2072��

global_step/Initializer/zerosConst*
_class
loc:@global_step*
value	B	 R *
dtype0	*
_output_shapes
: 
�
global_step
VariableV2*
dtype0	*
_output_shapes
: *
shared_name *
_class
loc:@global_step*
	container *
shape: 
�
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
use_locking(*
T0	*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
_class
loc:@global_step*
_output_shapes
: *
T0	
�
enqueue_input/fifo_queueFIFOQueueV2"/device:CPU:0*
_output_shapes
: *
component_types
2	*
shapes
: :� :*
shared_name *
capacity�*
	container 
m
enqueue_input/PlaceholderPlaceholder"/device:CPU:0*
shape:*
dtype0	*
_output_shapes
:
o
enqueue_input/Placeholder_1Placeholder"/device:CPU:0*
shape:*
dtype0*
_output_shapes
:
o
enqueue_input/Placeholder_2Placeholder"/device:CPU:0*
dtype0*
_output_shapes
:*
shape:
�
$enqueue_input/fifo_queue_EnqueueManyQueueEnqueueManyV2enqueue_input/fifo_queueenqueue_input/Placeholderenqueue_input/Placeholder_1enqueue_input/Placeholder_2"/device:CPU:0*
Tcomponents
2	*

timeout_ms���������
v
enqueue_input/fifo_queue_CloseQueueCloseV2enqueue_input/fifo_queue"/device:CPU:0*
cancel_pending_enqueues( 
x
 enqueue_input/fifo_queue_Close_1QueueCloseV2enqueue_input/fifo_queue"/device:CPU:0*
cancel_pending_enqueues(
m
enqueue_input/fifo_queue_SizeQueueSizeV2enqueue_input/fifo_queue"/device:CPU:0*
_output_shapes
: 
d
enqueue_input/sub/yConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : 
|
enqueue_input/subSubenqueue_input/fifo_queue_Sizeenqueue_input/sub/y"/device:CPU:0*
_output_shapes
: *
T0
h
enqueue_input/Maximum/xConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
|
enqueue_input/MaximumMaximumenqueue_input/Maximum/xenqueue_input/sub"/device:CPU:0*
_output_shapes
: *
T0
p
enqueue_input/CastCastenqueue_input/Maximum"/device:CPU:0*

SrcT0*
_output_shapes
: *

DstT0
g
enqueue_input/mul/yConst"/device:CPU:0*
valueB
 *o�:*
dtype0*
_output_shapes
: 
q
enqueue_input/mulMulenqueue_input/Castenqueue_input/mul/y"/device:CPU:0*
T0*
_output_shapes
: 
�
Menqueue_input/queue/enqueue_input/fifo_queuefraction_over_0_of_1000_full/tagsConst"/device:CPU:0*Y
valuePBN BHenqueue_input/queue/enqueue_input/fifo_queuefraction_over_0_of_1000_full*
dtype0*
_output_shapes
: 
�
Henqueue_input/queue/enqueue_input/fifo_queuefraction_over_0_of_1000_fullScalarSummaryMenqueue_input/queue/enqueue_input/fifo_queuefraction_over_0_of_1000_full/tagsenqueue_input/mul"/device:CPU:0*
T0*
_output_shapes
: 
i
fifo_queue_DequeueUpTo/nConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
fifo_queue_DequeueUpToQueueDequeueUpToV2enqueue_input/fifo_queuefifo_queue_DequeueUpTo/n"/device:CPU:0*J
_output_shapes8
6:���������:���������� :���������*
component_types
2	*

timeout_ms���������
�
2dnn/input_from_feature_columns/input_layer/x/ShapeShapefifo_queue_DequeueUpTo:1*
_output_shapes
:*
T0*
out_type0
�
@dnn/input_from_feature_columns/input_layer/x/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
Bdnn/input_from_feature_columns/input_layer/x/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
�
Bdnn/input_from_feature_columns/input_layer/x/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
:dnn/input_from_feature_columns/input_layer/x/strided_sliceStridedSlice2dnn/input_from_feature_columns/input_layer/x/Shape@dnn/input_from_feature_columns/input_layer/x/strided_slice/stackBdnn/input_from_feature_columns/input_layer/x/strided_slice/stack_1Bdnn/input_from_feature_columns/input_layer/x/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0

<dnn/input_from_feature_columns/input_layer/x/Reshape/shape/1Const*
value
B :� *
dtype0*
_output_shapes
: 
�
:dnn/input_from_feature_columns/input_layer/x/Reshape/shapePack:dnn/input_from_feature_columns/input_layer/x/strided_slice<dnn/input_from_feature_columns/input_layer/x/Reshape/shape/1*

axis *
N*
_output_shapes
:*
T0
�
4dnn/input_from_feature_columns/input_layer/x/ReshapeReshapefifo_queue_DequeueUpTo:1:dnn/input_from_feature_columns/input_layer/x/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:���������� 
~
<dnn/input_from_feature_columns/input_layer/concat/concat_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
1dnn/input_from_feature_columns/input_layer/concatIdentity4dnn/input_from_feature_columns/input_layer/x/Reshape*
T0*(
_output_shapes
:���������� 
�
@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shapeConst*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
valueB"      *
dtype0*
_output_shapes
:
�
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/minConst*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
valueB
 *0�*
dtype0*
_output_shapes
: 
�
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
valueB
 *0=
�
Hdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
� �*

seed *
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
seed2 
�
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes
: 
�
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0* 
_output_shapes
:
� �
�
:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min* 
_output_shapes
:
� �*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0
�
dnn/hiddenlayer_0/kernel/part_0
VariableV2*
dtype0* 
_output_shapes
:
� �*
shared_name *2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
	container *
shape:
� �
�
&dnn/hiddenlayer_0/kernel/part_0/AssignAssigndnn/hiddenlayer_0/kernel/part_0:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform* 
_output_shapes
:
� �*
use_locking(*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
validate_shape(
�
$dnn/hiddenlayer_0/kernel/part_0/readIdentitydnn/hiddenlayer_0/kernel/part_0*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0* 
_output_shapes
:
� �
�
/dnn/hiddenlayer_0/bias/part_0/Initializer/zerosConst*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
valueB�*    *
dtype0*
_output_shapes	
:�
�
dnn/hiddenlayer_0/bias/part_0
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0
�
$dnn/hiddenlayer_0/bias/part_0/AssignAssigndnn/hiddenlayer_0/bias/part_0/dnn/hiddenlayer_0/bias/part_0/Initializer/zeros*
use_locking(*
T0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
validate_shape(*
_output_shapes	
:�
�
"dnn/hiddenlayer_0/bias/part_0/readIdentitydnn/hiddenlayer_0/bias/part_0*
_output_shapes	
:�*
T0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0
u
dnn/hiddenlayer_0/kernelIdentity$dnn/hiddenlayer_0/kernel/part_0/read*
T0* 
_output_shapes
:
� �
�
dnn/hiddenlayer_0/MatMulMatMul1dnn/input_from_feature_columns/input_layer/concatdnn/hiddenlayer_0/kernel*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
l
dnn/hiddenlayer_0/biasIdentity"dnn/hiddenlayer_0/bias/part_0/read*
_output_shapes	
:�*
T0
�
dnn/hiddenlayer_0/BiasAddBiasAdddnn/hiddenlayer_0/MatMuldnn/hiddenlayer_0/bias*
T0*
data_formatNHWC*(
_output_shapes
:����������
l
dnn/hiddenlayer_0/ReluReludnn/hiddenlayer_0/BiasAdd*(
_output_shapes
:����������*
T0
[
dnn/zero_fraction/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
dnn/zero_fraction/EqualEqualdnn/hiddenlayer_0/Reludnn/zero_fraction/zero*(
_output_shapes
:����������*
T0
y
dnn/zero_fraction/CastCastdnn/zero_fraction/Equal*

SrcT0
*(
_output_shapes
:����������*

DstT0
h
dnn/zero_fraction/ConstConst*
dtype0*
_output_shapes
:*
valueB"       
�
dnn/zero_fraction/MeanMeandnn/zero_fraction/Castdnn/zero_fraction/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsConst*
_output_shapes
: *>
value5B3 B-dnn/dnn/hiddenlayer_0/fraction_of_zero_values*
dtype0
�
-dnn/dnn/hiddenlayer_0/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsdnn/zero_fraction/Mean*
_output_shapes
: *
T0
�
$dnn/dnn/hiddenlayer_0/activation/tagConst*1
value(B& B dnn/dnn/hiddenlayer_0/activation*
dtype0*
_output_shapes
: 
�
 dnn/dnn/hiddenlayer_0/activationHistogramSummary$dnn/dnn/hiddenlayer_0/activation/tagdnn/hiddenlayer_0/Relu*
T0*
_output_shapes
: 
�
@dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shapeConst*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
valueB"       *
dtype0*
_output_shapes
:
�
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/minConst*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
valueB
 *:��*
dtype0*
_output_shapes
: 
�
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/maxConst*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
valueB
 *:�>*
dtype0*
_output_shapes
: 
�
Hdnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shape*

seed *
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
seed2 *
dtype0*
_output_shapes
:	� 
�
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes
: 
�
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/sub*
_output_shapes
:	� *
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0
�
:dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes
:	� 
�
dnn/hiddenlayer_1/kernel/part_0
VariableV2*
shared_name *2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
	container *
shape:	� *
dtype0*
_output_shapes
:	� 
�
&dnn/hiddenlayer_1/kernel/part_0/AssignAssigndnn/hiddenlayer_1/kernel/part_0:dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform*
use_locking(*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
validate_shape(*
_output_shapes
:	� 
�
$dnn/hiddenlayer_1/kernel/part_0/readIdentitydnn/hiddenlayer_1/kernel/part_0*
_output_shapes
:	� *
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0
�
/dnn/hiddenlayer_1/bias/part_0/Initializer/zerosConst*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
valueB *    *
dtype0*
_output_shapes
: 
�
dnn/hiddenlayer_1/bias/part_0
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
	container 
�
$dnn/hiddenlayer_1/bias/part_0/AssignAssigndnn/hiddenlayer_1/bias/part_0/dnn/hiddenlayer_1/bias/part_0/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0
�
"dnn/hiddenlayer_1/bias/part_0/readIdentitydnn/hiddenlayer_1/bias/part_0*
T0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
_output_shapes
: 
t
dnn/hiddenlayer_1/kernelIdentity$dnn/hiddenlayer_1/kernel/part_0/read*
_output_shapes
:	� *
T0
�
dnn/hiddenlayer_1/MatMulMatMuldnn/hiddenlayer_0/Reludnn/hiddenlayer_1/kernel*
T0*'
_output_shapes
:��������� *
transpose_a( *
transpose_b( 
k
dnn/hiddenlayer_1/biasIdentity"dnn/hiddenlayer_1/bias/part_0/read*
_output_shapes
: *
T0
�
dnn/hiddenlayer_1/BiasAddBiasAdddnn/hiddenlayer_1/MatMuldnn/hiddenlayer_1/bias*
T0*
data_formatNHWC*'
_output_shapes
:��������� 
k
dnn/hiddenlayer_1/ReluReludnn/hiddenlayer_1/BiasAdd*
T0*'
_output_shapes
:��������� 
]
dnn/zero_fraction_1/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
dnn/zero_fraction_1/EqualEqualdnn/hiddenlayer_1/Reludnn/zero_fraction_1/zero*
T0*'
_output_shapes
:��������� 
|
dnn/zero_fraction_1/CastCastdnn/zero_fraction_1/Equal*

SrcT0
*'
_output_shapes
:��������� *

DstT0
j
dnn/zero_fraction_1/ConstConst*
_output_shapes
:*
valueB"       *
dtype0
�
dnn/zero_fraction_1/MeanMeandnn/zero_fraction_1/Castdnn/zero_fraction_1/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
2dnn/dnn/hiddenlayer_1/fraction_of_zero_values/tagsConst*>
value5B3 B-dnn/dnn/hiddenlayer_1/fraction_of_zero_values*
dtype0*
_output_shapes
: 
�
-dnn/dnn/hiddenlayer_1/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_1/fraction_of_zero_values/tagsdnn/zero_fraction_1/Mean*
T0*
_output_shapes
: 
�
$dnn/dnn/hiddenlayer_1/activation/tagConst*1
value(B& B dnn/dnn/hiddenlayer_1/activation*
dtype0*
_output_shapes
: 
�
 dnn/dnn/hiddenlayer_1/activationHistogramSummary$dnn/dnn/hiddenlayer_1/activation/tagdnn/hiddenlayer_1/Relu*
T0*
_output_shapes
: 
�
9dnn/logits/kernel/part_0/Initializer/random_uniform/shapeConst*+
_class!
loc:@dnn/logits/kernel/part_0*
valueB"       *
dtype0*
_output_shapes
:
�
7dnn/logits/kernel/part_0/Initializer/random_uniform/minConst*
_output_shapes
: *+
_class!
loc:@dnn/logits/kernel/part_0*
valueB
 *JQھ*
dtype0
�
7dnn/logits/kernel/part_0/Initializer/random_uniform/maxConst*+
_class!
loc:@dnn/logits/kernel/part_0*
valueB
 *JQ�>*
dtype0*
_output_shapes
: 
�
Adnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform9dnn/logits/kernel/part_0/Initializer/random_uniform/shape*
dtype0*
_output_shapes

: *

seed *
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
seed2 
�
7dnn/logits/kernel/part_0/Initializer/random_uniform/subSub7dnn/logits/kernel/part_0/Initializer/random_uniform/max7dnn/logits/kernel/part_0/Initializer/random_uniform/min*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes
: *
T0
�
7dnn/logits/kernel/part_0/Initializer/random_uniform/mulMulAdnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniform7dnn/logits/kernel/part_0/Initializer/random_uniform/sub*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

: 
�
3dnn/logits/kernel/part_0/Initializer/random_uniformAdd7dnn/logits/kernel/part_0/Initializer/random_uniform/mul7dnn/logits/kernel/part_0/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

: 
�
dnn/logits/kernel/part_0
VariableV2*
shape
: *
dtype0*
_output_shapes

: *
shared_name *+
_class!
loc:@dnn/logits/kernel/part_0*
	container 
�
dnn/logits/kernel/part_0/AssignAssigndnn/logits/kernel/part_03dnn/logits/kernel/part_0/Initializer/random_uniform*
validate_shape(*
_output_shapes

: *
use_locking(*
T0*+
_class!
loc:@dnn/logits/kernel/part_0
�
dnn/logits/kernel/part_0/readIdentitydnn/logits/kernel/part_0*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

: 
�
(dnn/logits/bias/part_0/Initializer/zerosConst*)
_class
loc:@dnn/logits/bias/part_0*
valueB*    *
dtype0*
_output_shapes
:
�
dnn/logits/bias/part_0
VariableV2*
shared_name *)
_class
loc:@dnn/logits/bias/part_0*
	container *
shape:*
dtype0*
_output_shapes
:
�
dnn/logits/bias/part_0/AssignAssigndnn/logits/bias/part_0(dnn/logits/bias/part_0/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*)
_class
loc:@dnn/logits/bias/part_0*
validate_shape(
�
dnn/logits/bias/part_0/readIdentitydnn/logits/bias/part_0*
T0*)
_class
loc:@dnn/logits/bias/part_0*
_output_shapes
:
e
dnn/logits/kernelIdentitydnn/logits/kernel/part_0/read*
T0*
_output_shapes

: 
�
dnn/logits/MatMulMatMuldnn/hiddenlayer_1/Reludnn/logits/kernel*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
]
dnn/logits/biasIdentitydnn/logits/bias/part_0/read*
_output_shapes
:*
T0
�
dnn/logits/BiasAddBiasAdddnn/logits/MatMuldnn/logits/bias*
data_formatNHWC*'
_output_shapes
:���������*
T0
]
dnn/zero_fraction_2/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
dnn/zero_fraction_2/EqualEqualdnn/logits/BiasAdddnn/zero_fraction_2/zero*
T0*'
_output_shapes
:���������
|
dnn/zero_fraction_2/CastCastdnn/zero_fraction_2/Equal*

SrcT0
*'
_output_shapes
:���������*

DstT0
j
dnn/zero_fraction_2/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
dnn/zero_fraction_2/MeanMeandnn/zero_fraction_2/Castdnn/zero_fraction_2/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
+dnn/dnn/logits/fraction_of_zero_values/tagsConst*
_output_shapes
: *7
value.B, B&dnn/dnn/logits/fraction_of_zero_values*
dtype0
�
&dnn/dnn/logits/fraction_of_zero_valuesScalarSummary+dnn/dnn/logits/fraction_of_zero_values/tagsdnn/zero_fraction_2/Mean*
T0*
_output_shapes
: 
w
dnn/dnn/logits/activation/tagConst**
value!B Bdnn/dnn/logits/activation*
dtype0*
_output_shapes
: 
�
dnn/dnn/logits/activationHistogramSummarydnn/dnn/logits/activation/tagdnn/logits/BiasAdd*
_output_shapes
: *
T0
s
!dnn/head/predictions/logits/ShapeShapednn/logits/BiasAdd*
T0*
out_type0*
_output_shapes
:
w
5dnn/head/predictions/logits/assert_rank_at_least/rankConst*
value	B :*
dtype0*
_output_shapes
: 
g
_dnn/head/predictions/logits/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp
X
Pdnn/head/predictions/logits/assert_rank_at_least/static_checks_determined_all_okNoOp
n
dnn/head/predictions/logisticSigmoiddnn/logits/BiasAdd*'
_output_shapes
:���������*
T0
r
dnn/head/predictions/zeros_like	ZerosLikednn/logits/BiasAdd*
T0*'
_output_shapes
:���������
u
*dnn/head/predictions/two_class_logits/axisConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
%dnn/head/predictions/two_class_logitsConcatV2dnn/head/predictions/zeros_likednn/logits/BiasAdd*dnn/head/predictions/two_class_logits/axis*
N*'
_output_shapes
:���������*

Tidx0*
T0
�
"dnn/head/predictions/probabilitiesSoftmax%dnn/head/predictions/two_class_logits*
T0*'
_output_shapes
:���������
s
(dnn/head/predictions/class_ids/dimensionConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
dnn/head/predictions/class_idsArgMax%dnn/head/predictions/two_class_logits(dnn/head/predictions/class_ids/dimension*
output_type0	*#
_output_shapes
:���������*

Tidx0*
T0
n
#dnn/head/predictions/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
dnn/head/predictions/ExpandDims
ExpandDimsdnn/head/predictions/class_ids#dnn/head/predictions/ExpandDims/dim*

Tdim0*
T0	*'
_output_shapes
:���������
�
 dnn/head/predictions/str_classesAsStringdnn/head/predictions/ExpandDims*

fill *

scientific( *
width���������*'
_output_shapes
:���������*
shortest( *
	precision���������*
T0	
m
dnn/head/labels/ShapeShapefifo_queue_DequeueUpTo:2*
T0*
out_type0*
_output_shapes
:
i
dnn/head/labels/Shape_1Shapednn/logits/BiasAdd*
T0*
out_type0*
_output_shapes
:
k
)dnn/head/labels/assert_rank_at_least/rankConst*
value	B :*
dtype0*
_output_shapes
: 
�
*dnn/head/labels/assert_rank_at_least/ShapeShapefifo_queue_DequeueUpTo:2*
out_type0*
_output_shapes
:*
T0
[
Sdnn/head/labels/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp
L
Ddnn/head/labels/assert_rank_at_least/static_checks_determined_all_okNoOp
�
#dnn/head/labels/strided_slice/stackConstE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok*
dtype0*
_output_shapes
:*
valueB: 
�
%dnn/head/labels/strided_slice/stack_1ConstE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok*
_output_shapes
:*
valueB:
���������*
dtype0
�
%dnn/head/labels/strided_slice/stack_2ConstE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok*
_output_shapes
:*
valueB:*
dtype0
�
dnn/head/labels/strided_sliceStridedSlicednn/head/labels/Shape_1#dnn/head/labels/strided_slice/stack%dnn/head/labels/strided_slice/stack_1%dnn/head/labels/strided_slice/stack_2*
new_axis_mask *
end_mask *
_output_shapes
:*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask 
�
dnn/head/labels/concat/values_1ConstE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok*
valueB:*
dtype0*
_output_shapes
:
�
dnn/head/labels/concat/axisConstE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok*
value	B : *
dtype0*
_output_shapes
: 
�
dnn/head/labels/concatConcatV2dnn/head/labels/strided_slicednn/head/labels/concat/values_1dnn/head/labels/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0

"dnn/head/labels/assert_equal/EqualEqualdnn/head/labels/concatdnn/head/labels/Shape*
T0*
_output_shapes
:
�
"dnn/head/labels/assert_equal/ConstConstE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok*
_output_shapes
:*
valueB: *
dtype0
�
 dnn/head/labels/assert_equal/AllAll"dnn/head/labels/assert_equal/Equal"dnn/head/labels/assert_equal/Const*
_output_shapes
: *
	keep_dims( *

Tidx0
�
)dnn/head/labels/assert_equal/Assert/ConstConstE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok*(
valueB Bexpected_labels_shape: *
dtype0*
_output_shapes
: 
�
+dnn/head/labels/assert_equal/Assert/Const_1ConstE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok*
valueB Blabels_shape: *
dtype0*
_output_shapes
: 
�
1dnn/head/labels/assert_equal/Assert/Assert/data_0ConstE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok*(
valueB Bexpected_labels_shape: *
dtype0*
_output_shapes
: 
�
1dnn/head/labels/assert_equal/Assert/Assert/data_2ConstE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok*
valueB Blabels_shape: *
dtype0*
_output_shapes
: 
�
*dnn/head/labels/assert_equal/Assert/AssertAssert dnn/head/labels/assert_equal/All1dnn/head/labels/assert_equal/Assert/Assert/data_0dnn/head/labels/concat1dnn/head/labels/assert_equal/Assert/Assert/data_2dnn/head/labels/Shape*
T
2*
	summarize
�
dnn/head/labelsIdentityfifo_queue_DequeueUpTo:2+^dnn/head/labels/assert_equal/Assert/AssertE^dnn/head/labels/assert_rank_at_least/static_checks_determined_all_ok*
T0*'
_output_shapes
:���������
j
dnn/head/ToFloatCastdnn/head/labels*

SrcT0*'
_output_shapes
:���������*

DstT0
`
dnn/head/assert_range/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *   @
�
&dnn/head/assert_range/assert_less/LessLessdnn/head/ToFloatdnn/head/assert_range/Const*
T0*'
_output_shapes
:���������
x
'dnn/head/assert_range/assert_less/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
%dnn/head/assert_range/assert_less/AllAll&dnn/head/assert_range/assert_less/Less'dnn/head/assert_range/assert_less/Const*
_output_shapes
: *
	keep_dims( *

Tidx0
�
.dnn/head/assert_range/assert_less/Assert/ConstConst*
dtype0*
_output_shapes
: *+
value"B  BLabel IDs must < n_classes
�
0dnn/head/assert_range/assert_less/Assert/Const_1Const*
_output_shapes
: *;
value2B0 B*Condition x < y did not hold element-wise:*
dtype0
�
0dnn/head/assert_range/assert_less/Assert/Const_2Const**
value!B Bx (dnn/head/ToFloat:0) = *
dtype0*
_output_shapes
: 
�
0dnn/head/assert_range/assert_less/Assert/Const_3Const*
dtype0*
_output_shapes
: *5
value,B* B$y (dnn/head/assert_range/Const:0) = 
�
;dnn/head/assert_range/assert_less/Assert/AssertGuard/SwitchSwitch%dnn/head/assert_range/assert_less/All%dnn/head/assert_range/assert_less/All*
_output_shapes
: : *
T0

�
=dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_tIdentity=dnn/head/assert_range/assert_less/Assert/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
�
=dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_fIdentity;dnn/head/assert_range/assert_less/Assert/AssertGuard/Switch*
T0
*
_output_shapes
: 
�
<dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_idIdentity%dnn/head/assert_range/assert_less/All*
T0
*
_output_shapes
: 
�
9dnn/head/assert_range/assert_less/Assert/AssertGuard/NoOpNoOp>^dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_t
�
Gdnn/head/assert_range/assert_less/Assert/AssertGuard/control_dependencyIdentity=dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_t:^dnn/head/assert_range/assert_less/Assert/AssertGuard/NoOp*P
_classF
DBloc:@dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_t*
_output_shapes
: *
T0

�
Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_0Const>^dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f*+
value"B  BLabel IDs must < n_classes*
dtype0*
_output_shapes
: 
�
Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_1Const>^dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f*;
value2B0 B*Condition x < y did not hold element-wise:*
dtype0*
_output_shapes
: 
�
Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_2Const>^dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f**
value!B Bx (dnn/head/ToFloat:0) = *
dtype0*
_output_shapes
: 
�
Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_4Const>^dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f*5
value,B* B$y (dnn/head/assert_range/Const:0) = *
dtype0*
_output_shapes
: 
�
;dnn/head/assert_range/assert_less/Assert/AssertGuard/AssertAssertBdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/SwitchBdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_0Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_1Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_2Ddnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/Switch_1Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_4Ddnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/Switch_2*
T

2*
	summarize
�
Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/SwitchSwitch%dnn/head/assert_range/assert_less/All<dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_id*
T0
*8
_class.
,*loc:@dnn/head/assert_range/assert_less/All*
_output_shapes
: : 
�
Ddnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/Switch_1Switchdnn/head/ToFloat<dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_id*:
_output_shapes(
&:���������:���������*
T0*#
_class
loc:@dnn/head/ToFloat
�
Ddnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/Switch_2Switchdnn/head/assert_range/Const<dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_id*
T0*.
_class$
" loc:@dnn/head/assert_range/Const*
_output_shapes
: : 
�
Idnn/head/assert_range/assert_less/Assert/AssertGuard/control_dependency_1Identity=dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f<^dnn/head/assert_range/assert_less/Assert/AssertGuard/Assert*
T0
*P
_classF
DBloc:@dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f*
_output_shapes
: 
�
:dnn/head/assert_range/assert_less/Assert/AssertGuard/MergeMergeIdnn/head/assert_range/assert_less/Assert/AssertGuard/control_dependency_1Gdnn/head/assert_range/assert_less/Assert/AssertGuard/control_dependency*
N*
_output_shapes
: : *
T0

t
/dnn/head/assert_range/assert_non_negative/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Ednn/head/assert_range/assert_non_negative/assert_less_equal/LessEqual	LessEqual/dnn/head/assert_range/assert_non_negative/Constdnn/head/ToFloat*'
_output_shapes
:���������*
T0
�
Adnn/head/assert_range/assert_non_negative/assert_less_equal/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
?dnn/head/assert_range/assert_non_negative/assert_less_equal/AllAllEdnn/head/assert_range/assert_non_negative/assert_less_equal/LessEqualAdnn/head/assert_range/assert_non_negative/assert_less_equal/Const*
_output_shapes
: *
	keep_dims( *

Tidx0
�
Hdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/ConstConst*$
valueB BLabel IDs must >= 0*
dtype0*
_output_shapes
: 
�
Jdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/Const_1Const*
_output_shapes
: *<
value3B1 B+Condition x >= 0 did not hold element-wise:*
dtype0
�
Jdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/Const_2Const**
value!B Bx (dnn/head/ToFloat:0) = *
dtype0*
_output_shapes
: 
�
Udnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/SwitchSwitch?dnn/head/assert_range/assert_non_negative/assert_less_equal/All?dnn/head/assert_range/assert_non_negative/assert_less_equal/All*
T0
*
_output_shapes
: : 
�
Wdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_tIdentityWdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
�
Wdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_fIdentityUdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Switch*
T0
*
_output_shapes
: 
�
Vdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_idIdentity?dnn/head/assert_range/assert_non_negative/assert_less_equal/All*
T0
*
_output_shapes
: 
�
Sdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOpNoOpX^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_t
�
adnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/control_dependencyIdentityWdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_tT^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOp*
_output_shapes
: *
T0
*j
_class`
^\loc:@dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_t
�
\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0ConstX^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_f*$
valueB BLabel IDs must >= 0*
dtype0*
_output_shapes
: 
�
\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1ConstX^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_f*<
value3B1 B+Condition x >= 0 did not hold element-wise:*
dtype0*
_output_shapes
: 
�
\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2ConstX^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_f*
dtype0*
_output_shapes
: **
value!B Bx (dnn/head/ToFloat:0) = 
�
Udnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/AssertAssert\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/Switch\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/Switch_1*
T
2*
	summarize
�
\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/SwitchSwitch?dnn/head/assert_range/assert_non_negative/assert_less_equal/AllVdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_id*
_output_shapes
: : *
T0
*R
_classH
FDloc:@dnn/head/assert_range/assert_non_negative/assert_less_equal/All
�
^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/Switch_1Switchdnn/head/ToFloatVdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_id*:
_output_shapes(
&:���������:���������*
T0*#
_class
loc:@dnn/head/ToFloat
�
cdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/control_dependency_1IdentityWdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_fV^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert*
T0
*j
_class`
^\loc:@dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_f*
_output_shapes
: 
�
Tdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/MergeMergecdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/control_dependency_1adnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/control_dependency*
T0
*
N*
_output_shapes
: : 
�
dnn/head/assert_range/IdentityIdentitydnn/head/ToFloat;^dnn/head/assert_range/assert_less/Assert/AssertGuard/MergeU^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Merge*'
_output_shapes
:���������*
T0
t
!dnn/head/logistic_loss/zeros_like	ZerosLikednn/logits/BiasAdd*
T0*'
_output_shapes
:���������
�
#dnn/head/logistic_loss/GreaterEqualGreaterEqualdnn/logits/BiasAdd!dnn/head/logistic_loss/zeros_like*
T0*'
_output_shapes
:���������
�
dnn/head/logistic_loss/SelectSelect#dnn/head/logistic_loss/GreaterEqualdnn/logits/BiasAdd!dnn/head/logistic_loss/zeros_like*'
_output_shapes
:���������*
T0
g
dnn/head/logistic_loss/NegNegdnn/logits/BiasAdd*'
_output_shapes
:���������*
T0
�
dnn/head/logistic_loss/Select_1Select#dnn/head/logistic_loss/GreaterEqualdnn/head/logistic_loss/Negdnn/logits/BiasAdd*'
_output_shapes
:���������*
T0
�
dnn/head/logistic_loss/mulMuldnn/logits/BiasAdddnn/head/assert_range/Identity*'
_output_shapes
:���������*
T0
�
dnn/head/logistic_loss/subSubdnn/head/logistic_loss/Selectdnn/head/logistic_loss/mul*
T0*'
_output_shapes
:���������
t
dnn/head/logistic_loss/ExpExpdnn/head/logistic_loss/Select_1*
T0*'
_output_shapes
:���������
s
dnn/head/logistic_loss/Log1pLog1pdnn/head/logistic_loss/Exp*
T0*'
_output_shapes
:���������
�
dnn/head/logistic_lossAdddnn/head/logistic_loss/subdnn/head/logistic_loss/Log1p*'
_output_shapes
:���������*
T0
x
3dnn/head/weighted_loss/assert_broadcastable/weightsConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
|
9dnn/head/weighted_loss/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
z
8dnn/head/weighted_loss/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
�
8dnn/head/weighted_loss/assert_broadcastable/values/shapeShapednn/head/logistic_loss*
T0*
out_type0*
_output_shapes
:
y
7dnn/head/weighted_loss/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
O
Gdnn/head/weighted_loss/assert_broadcastable/static_scalar_check_successNoOp
�
"dnn/head/weighted_loss/ToFloat_1/xConstH^dnn/head/weighted_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
dnn/head/weighted_loss/MulMuldnn/head/logistic_loss"dnn/head/weighted_loss/ToFloat_1/x*
T0*'
_output_shapes
:���������
�
dnn/head/weighted_loss/ConstConstH^dnn/head/weighted_loss/assert_broadcastable/static_scalar_check_success*
valueB"       *
dtype0*
_output_shapes
:
�
dnn/head/weighted_loss/SumSumdnn/head/weighted_loss/Muldnn/head/weighted_loss/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
z
5dnn/head/metrics/label/mean/broadcast_weights/weightsConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Pdnn/head/metrics/label/mean/broadcast_weights/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Odnn/head/metrics/label/mean/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
�
Odnn/head/metrics/label/mean/broadcast_weights/assert_broadcastable/values/shapeShapednn/head/assert_range/Identity*
T0*
out_type0*
_output_shapes
:
�
Ndnn/head/metrics/label/mean/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
f
^dnn/head/metrics/label/mean/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
�
=dnn/head/metrics/label/mean/broadcast_weights/ones_like/ShapeShapednn/head/assert_range/Identity_^dnn/head/metrics/label/mean/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
�
=dnn/head/metrics/label/mean/broadcast_weights/ones_like/ConstConst_^dnn/head/metrics/label/mean/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
7dnn/head/metrics/label/mean/broadcast_weights/ones_likeFill=dnn/head/metrics/label/mean/broadcast_weights/ones_like/Shape=dnn/head/metrics/label/mean/broadcast_weights/ones_like/Const*
T0*

index_type0*'
_output_shapes
:���������
�
-dnn/head/metrics/label/mean/broadcast_weightsMul5dnn/head/metrics/label/mean/broadcast_weights/weights7dnn/head/metrics/label/mean/broadcast_weights/ones_like*'
_output_shapes
:���������*
T0
�
3dnn/head/metrics/label/mean/total/Initializer/zerosConst*4
_class*
(&loc:@dnn/head/metrics/label/mean/total*
valueB
 *    *
dtype0*
_output_shapes
: 
�
!dnn/head/metrics/label/mean/total
VariableV2*
dtype0*
_output_shapes
: *
shared_name *4
_class*
(&loc:@dnn/head/metrics/label/mean/total*
	container *
shape: 
�
(dnn/head/metrics/label/mean/total/AssignAssign!dnn/head/metrics/label/mean/total3dnn/head/metrics/label/mean/total/Initializer/zeros*
_output_shapes
: *
use_locking(*
T0*4
_class*
(&loc:@dnn/head/metrics/label/mean/total*
validate_shape(
�
&dnn/head/metrics/label/mean/total/readIdentity!dnn/head/metrics/label/mean/total*
T0*4
_class*
(&loc:@dnn/head/metrics/label/mean/total*
_output_shapes
: 
�
3dnn/head/metrics/label/mean/count/Initializer/zerosConst*4
_class*
(&loc:@dnn/head/metrics/label/mean/count*
valueB
 *    *
dtype0*
_output_shapes
: 
�
!dnn/head/metrics/label/mean/count
VariableV2*4
_class*
(&loc:@dnn/head/metrics/label/mean/count*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
�
(dnn/head/metrics/label/mean/count/AssignAssign!dnn/head/metrics/label/mean/count3dnn/head/metrics/label/mean/count/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@dnn/head/metrics/label/mean/count*
validate_shape(*
_output_shapes
: 
�
&dnn/head/metrics/label/mean/count/readIdentity!dnn/head/metrics/label/mean/count*
_output_shapes
: *
T0*4
_class*
(&loc:@dnn/head/metrics/label/mean/count
�
Rdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/shapeShape-dnn/head/metrics/label/mean/broadcast_weights*
T0*
out_type0*
_output_shapes
:
�
Qdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/rankConst*
_output_shapes
: *
value	B :*
dtype0
�
Qdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/shapeShapednn/head/assert_range/Identity*
T0*
out_type0*
_output_shapes
:
�
Pdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
�
Pdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalar/xConst*
value	B : *
dtype0*
_output_shapes
: 
�
Ndnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalarEqualPdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalar/xQdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/rank*
_output_shapes
: *
T0
�
Zdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/SwitchSwitchNdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalarNdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
�
\dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_tIdentity\dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch:1*
_output_shapes
: *
T0

�
\dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_fIdentityZdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch*
_output_shapes
: *
T0

�
[dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_idIdentityNdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: 
�
\dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1SwitchNdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalar[dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*a
_classW
USloc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalar*
_output_shapes
: : *
T0

�
zdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqual�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1*
T0*
_output_shapes
: 
�
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchSwitchPdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/rank[dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*c
_classY
WUloc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/rank*
_output_shapes
: : *
T0
�
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1SwitchQdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/rank[dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*d
_classZ
XVloc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/rank*
_output_shapes
: : 
�
tdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/SwitchSwitchzdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankzdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: : 
�
vdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_tIdentityvdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1*
_output_shapes
: *
T0

�
vdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_fIdentitytdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch*
T0
*
_output_shapes
: 
�
udnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_idIdentityzdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: *
T0

�
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConstw^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
���������*
dtype0*
_output_shapes
: 
�
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDims�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim*

Tdim0*
T0*
_output_shapes

:
�
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchSwitchQdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/shape[dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*d
_classZ
XVloc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/shape* 
_output_shapes
::
�
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1Switch�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switchudnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*d
_classZ
XVloc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/shape* 
_output_shapes
::
�
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeConstw^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB"      *
dtype0*
_output_shapes
:
�
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConstw^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
_output_shapes
: *
value	B :*
dtype0
�
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFill�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const*
_output_shapes

:*
T0*

index_type0
�
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConstw^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
�
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis*

Tidx0*
T0*
N*
_output_shapes

:
�
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConstw^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
���������*
dtype0*
_output_shapes
: 
�
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDims�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim*
T0*
_output_shapes

:*

Tdim0
�
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchSwitchRdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/shape[dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*e
_class[
YWloc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/shape* 
_output_shapes
::
�
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1Switch�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switchudnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*e
_class[
YWloc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/shape* 
_output_shapes
::
�
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperation�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat*
validate_indices(*
T0*<
_output_shapes*
(:���������:���������:*
set_operationa-b
�
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSize�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1*
T0*
out_type0*
_output_shapes
: 
�
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConstw^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B : *
dtype0*
_output_shapes
: 
�
~dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqual�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims*
_output_shapes
: *
T0
�
vdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Switchzdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankudnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0
*�
_class�
�loc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: : 
�
sdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeMergevdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1~dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims*
T0
*
N*
_output_shapes
: : 
�
Ydnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/MergeMergesdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1:1*
T0
*
N*
_output_shapes
: : 
�
Jdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/ConstConst*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
�
Ldnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/Const_1Const*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
�
Ldnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/Const_2Const*@
value7B5 B/dnn/head/metrics/label/mean/broadcast_weights:0*
dtype0*
_output_shapes
: 
�
Ldnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/Const_3Const*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
�
Ldnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/Const_4Const*1
value(B& B dnn/head/assert_range/Identity:0*
dtype0*
_output_shapes
: 
�
Ldnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/Const_5Const*
dtype0*
_output_shapes
: *
valueB B
is_scalar=
�
Wdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/SwitchSwitchYdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/MergeYdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: : 
�
Ydnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_tIdentityYdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Switch:1*
_output_shapes
: *
T0

�
Ydnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_fIdentityWdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Switch*
_output_shapes
: *
T0

�
Xdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_idIdentityYdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: *
T0

�
Udnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/NoOpNoOpZ^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t
�
cdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependencyIdentityYdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_tV^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/NoOp*
_output_shapes
: *
T0
*l
_classb
`^loc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t
�
^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0ConstZ^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
�
^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1ConstZ^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
�
^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2ConstZ^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
dtype0*
_output_shapes
: *@
value7B5 B/dnn/head/metrics/label/mean/broadcast_weights:0
�
^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4ConstZ^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
�
^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5ConstZ^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: *1
value(B& B dnn/head/assert_range/Identity:0*
dtype0
�
^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7ConstZ^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
�
Wdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/AssertAssert^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2`dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5`dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7`dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3*
T
2	
*
	summarize
�
^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/SwitchSwitchYdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/MergeXdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0
*l
_classb
`^loc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: : 
�
`dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1SwitchRdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/shapeXdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0*e
_class[
YWloc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/shape* 
_output_shapes
::
�
`dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2SwitchQdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/shapeXdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0*d
_classZ
XVloc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/shape* 
_output_shapes
::
�
`dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3SwitchNdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalarXdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
_output_shapes
: : *
T0
*a
_classW
USloc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalar
�
ednn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency_1IdentityYdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_fX^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert*
_output_shapes
: *
T0
*l
_classb
`^loc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f
�
Vdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/MergeMergeednn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency_1cdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency*
N*
_output_shapes
: : *
T0

�
?dnn/head/metrics/label/mean/broadcast_weights_1/ones_like/ShapeShapednn/head/assert_range/IdentityW^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Merge*
out_type0*
_output_shapes
:*
T0
�
?dnn/head/metrics/label/mean/broadcast_weights_1/ones_like/ConstConstW^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Merge*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
9dnn/head/metrics/label/mean/broadcast_weights_1/ones_likeFill?dnn/head/metrics/label/mean/broadcast_weights_1/ones_like/Shape?dnn/head/metrics/label/mean/broadcast_weights_1/ones_like/Const*
T0*

index_type0*'
_output_shapes
:���������
�
/dnn/head/metrics/label/mean/broadcast_weights_1Mul-dnn/head/metrics/label/mean/broadcast_weights9dnn/head/metrics/label/mean/broadcast_weights_1/ones_like*'
_output_shapes
:���������*
T0
�
dnn/head/metrics/label/mean/MulMuldnn/head/assert_range/Identity/dnn/head/metrics/label/mean/broadcast_weights_1*
T0*'
_output_shapes
:���������
r
!dnn/head/metrics/label/mean/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
dnn/head/metrics/label/mean/SumSum/dnn/head/metrics/label/mean/broadcast_weights_1!dnn/head/metrics/label/mean/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
t
#dnn/head/metrics/label/mean/Const_1Const*
valueB"       *
dtype0*
_output_shapes
:
�
!dnn/head/metrics/label/mean/Sum_1Sumdnn/head/metrics/label/mean/Mul#dnn/head/metrics/label/mean/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
%dnn/head/metrics/label/mean/AssignAdd	AssignAdd!dnn/head/metrics/label/mean/total!dnn/head/metrics/label/mean/Sum_1*
_output_shapes
: *
use_locking( *
T0*4
_class*
(&loc:@dnn/head/metrics/label/mean/total
�
'dnn/head/metrics/label/mean/AssignAdd_1	AssignAdd!dnn/head/metrics/label/mean/countdnn/head/metrics/label/mean/Sum ^dnn/head/metrics/label/mean/Mul*
T0*4
_class*
(&loc:@dnn/head/metrics/label/mean/count*
_output_shapes
: *
use_locking( 
�
#dnn/head/metrics/label/mean/truedivRealDiv&dnn/head/metrics/label/mean/total/read&dnn/head/metrics/label/mean/count/read*
T0*
_output_shapes
: 
k
&dnn/head/metrics/label/mean/zeros_likeConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
#dnn/head/metrics/label/mean/GreaterGreater&dnn/head/metrics/label/mean/count/read&dnn/head/metrics/label/mean/zeros_like*
T0*
_output_shapes
: 
�
!dnn/head/metrics/label/mean/valueSelect#dnn/head/metrics/label/mean/Greater#dnn/head/metrics/label/mean/truediv&dnn/head/metrics/label/mean/zeros_like*
T0*
_output_shapes
: 
�
%dnn/head/metrics/label/mean/truediv_1RealDiv%dnn/head/metrics/label/mean/AssignAdd'dnn/head/metrics/label/mean/AssignAdd_1*
T0*
_output_shapes
: 
m
(dnn/head/metrics/label/mean/zeros_like_1Const*
valueB
 *    *
dtype0*
_output_shapes
: 
�
%dnn/head/metrics/label/mean/Greater_1Greater'dnn/head/metrics/label/mean/AssignAdd_1(dnn/head/metrics/label/mean/zeros_like_1*
_output_shapes
: *
T0
�
%dnn/head/metrics/label/mean/update_opSelect%dnn/head/metrics/label/mean/Greater_1%dnn/head/metrics/label/mean/truediv_1(dnn/head/metrics/label/mean/zeros_like_1*
T0*
_output_shapes
: 
�
5dnn/head/metrics/average_loss/total/Initializer/zerosConst*6
_class,
*(loc:@dnn/head/metrics/average_loss/total*
valueB
 *    *
dtype0*
_output_shapes
: 
�
#dnn/head/metrics/average_loss/total
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *6
_class,
*(loc:@dnn/head/metrics/average_loss/total*
	container 
�
*dnn/head/metrics/average_loss/total/AssignAssign#dnn/head/metrics/average_loss/total5dnn/head/metrics/average_loss/total/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*6
_class,
*(loc:@dnn/head/metrics/average_loss/total
�
(dnn/head/metrics/average_loss/total/readIdentity#dnn/head/metrics/average_loss/total*
T0*6
_class,
*(loc:@dnn/head/metrics/average_loss/total*
_output_shapes
: 
�
5dnn/head/metrics/average_loss/count/Initializer/zerosConst*6
_class,
*(loc:@dnn/head/metrics/average_loss/count*
valueB
 *    *
dtype0*
_output_shapes
: 
�
#dnn/head/metrics/average_loss/count
VariableV2*
shared_name *6
_class,
*(loc:@dnn/head/metrics/average_loss/count*
	container *
shape: *
dtype0*
_output_shapes
: 
�
*dnn/head/metrics/average_loss/count/AssignAssign#dnn/head/metrics/average_loss/count5dnn/head/metrics/average_loss/count/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@dnn/head/metrics/average_loss/count*
validate_shape(*
_output_shapes
: 
�
(dnn/head/metrics/average_loss/count/readIdentity#dnn/head/metrics/average_loss/count*
T0*6
_class,
*(loc:@dnn/head/metrics/average_loss/count*
_output_shapes
: 
h
#dnn/head/metrics/average_loss/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Rdnn/head/metrics/average_loss/broadcast_weights/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Qdnn/head/metrics/average_loss/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
�
Qdnn/head/metrics/average_loss/broadcast_weights/assert_broadcastable/values/shapeShapednn/head/logistic_loss*
T0*
out_type0*
_output_shapes
:
�
Pdnn/head/metrics/average_loss/broadcast_weights/assert_broadcastable/values/rankConst*
_output_shapes
: *
value	B :*
dtype0
h
`dnn/head/metrics/average_loss/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
�
?dnn/head/metrics/average_loss/broadcast_weights/ones_like/ShapeShapednn/head/logistic_lossa^dnn/head/metrics/average_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
�
?dnn/head/metrics/average_loss/broadcast_weights/ones_like/ConstConsta^dnn/head/metrics/average_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
9dnn/head/metrics/average_loss/broadcast_weights/ones_likeFill?dnn/head/metrics/average_loss/broadcast_weights/ones_like/Shape?dnn/head/metrics/average_loss/broadcast_weights/ones_like/Const*
T0*

index_type0*'
_output_shapes
:���������
�
/dnn/head/metrics/average_loss/broadcast_weightsMul#dnn/head/metrics/average_loss/Const9dnn/head/metrics/average_loss/broadcast_weights/ones_like*'
_output_shapes
:���������*
T0
�
!dnn/head/metrics/average_loss/MulMuldnn/head/logistic_loss/dnn/head/metrics/average_loss/broadcast_weights*'
_output_shapes
:���������*
T0
v
%dnn/head/metrics/average_loss/Const_1Const*
valueB"       *
dtype0*
_output_shapes
:
�
!dnn/head/metrics/average_loss/SumSum/dnn/head/metrics/average_loss/broadcast_weights%dnn/head/metrics/average_loss/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
v
%dnn/head/metrics/average_loss/Const_2Const*
dtype0*
_output_shapes
:*
valueB"       
�
#dnn/head/metrics/average_loss/Sum_1Sum!dnn/head/metrics/average_loss/Mul%dnn/head/metrics/average_loss/Const_2*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
'dnn/head/metrics/average_loss/AssignAdd	AssignAdd#dnn/head/metrics/average_loss/total#dnn/head/metrics/average_loss/Sum_1*
use_locking( *
T0*6
_class,
*(loc:@dnn/head/metrics/average_loss/total*
_output_shapes
: 
�
)dnn/head/metrics/average_loss/AssignAdd_1	AssignAdd#dnn/head/metrics/average_loss/count!dnn/head/metrics/average_loss/Sum"^dnn/head/metrics/average_loss/Mul*6
_class,
*(loc:@dnn/head/metrics/average_loss/count*
_output_shapes
: *
use_locking( *
T0
�
%dnn/head/metrics/average_loss/truedivRealDiv(dnn/head/metrics/average_loss/total/read(dnn/head/metrics/average_loss/count/read*
T0*
_output_shapes
: 
m
(dnn/head/metrics/average_loss/zeros_likeConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
%dnn/head/metrics/average_loss/GreaterGreater(dnn/head/metrics/average_loss/count/read(dnn/head/metrics/average_loss/zeros_like*
T0*
_output_shapes
: 
�
#dnn/head/metrics/average_loss/valueSelect%dnn/head/metrics/average_loss/Greater%dnn/head/metrics/average_loss/truediv(dnn/head/metrics/average_loss/zeros_like*
T0*
_output_shapes
: 
�
'dnn/head/metrics/average_loss/truediv_1RealDiv'dnn/head/metrics/average_loss/AssignAdd)dnn/head/metrics/average_loss/AssignAdd_1*
T0*
_output_shapes
: 
o
*dnn/head/metrics/average_loss/zeros_like_1Const*
valueB
 *    *
dtype0*
_output_shapes
: 
�
'dnn/head/metrics/average_loss/Greater_1Greater)dnn/head/metrics/average_loss/AssignAdd_1*dnn/head/metrics/average_loss/zeros_like_1*
_output_shapes
: *
T0
�
'dnn/head/metrics/average_loss/update_opSelect'dnn/head/metrics/average_loss/Greater_1'dnn/head/metrics/average_loss/truediv_1*dnn/head/metrics/average_loss/zeros_like_1*
_output_shapes
: *
T0
[
dnn/head/metrics/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 

dnn/head/metrics/CastCastdnn/head/predictions/ExpandDims*

SrcT0	*'
_output_shapes
:���������*

DstT0
�
dnn/head/metrics/EqualEqualdnn/head/metrics/Castdnn/head/assert_range/Identity*
T0*'
_output_shapes
:���������
y
dnn/head/metrics/ToFloatCastdnn/head/metrics/Equal*'
_output_shapes
:���������*

DstT0*

SrcT0

�
1dnn/head/metrics/accuracy/total/Initializer/zerosConst*2
_class(
&$loc:@dnn/head/metrics/accuracy/total*
valueB
 *    *
dtype0*
_output_shapes
: 
�
dnn/head/metrics/accuracy/total
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *2
_class(
&$loc:@dnn/head/metrics/accuracy/total
�
&dnn/head/metrics/accuracy/total/AssignAssigndnn/head/metrics/accuracy/total1dnn/head/metrics/accuracy/total/Initializer/zeros*
use_locking(*
T0*2
_class(
&$loc:@dnn/head/metrics/accuracy/total*
validate_shape(*
_output_shapes
: 
�
$dnn/head/metrics/accuracy/total/readIdentitydnn/head/metrics/accuracy/total*
_output_shapes
: *
T0*2
_class(
&$loc:@dnn/head/metrics/accuracy/total
�
1dnn/head/metrics/accuracy/count/Initializer/zerosConst*2
_class(
&$loc:@dnn/head/metrics/accuracy/count*
valueB
 *    *
dtype0*
_output_shapes
: 
�
dnn/head/metrics/accuracy/count
VariableV2*
dtype0*
_output_shapes
: *
shared_name *2
_class(
&$loc:@dnn/head/metrics/accuracy/count*
	container *
shape: 
�
&dnn/head/metrics/accuracy/count/AssignAssigndnn/head/metrics/accuracy/count1dnn/head/metrics/accuracy/count/Initializer/zeros*
_output_shapes
: *
use_locking(*
T0*2
_class(
&$loc:@dnn/head/metrics/accuracy/count*
validate_shape(
�
$dnn/head/metrics/accuracy/count/readIdentitydnn/head/metrics/accuracy/count*
T0*2
_class(
&$loc:@dnn/head/metrics/accuracy/count*
_output_shapes
: 
�
Ndnn/head/metrics/accuracy/broadcast_weights/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Mdnn/head/metrics/accuracy/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
�
Mdnn/head/metrics/accuracy/broadcast_weights/assert_broadcastable/values/shapeShapednn/head/metrics/ToFloat*
T0*
out_type0*
_output_shapes
:
�
Ldnn/head/metrics/accuracy/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
d
\dnn/head/metrics/accuracy/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
�
;dnn/head/metrics/accuracy/broadcast_weights/ones_like/ShapeShapednn/head/metrics/ToFloat]^dnn/head/metrics/accuracy/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
�
;dnn/head/metrics/accuracy/broadcast_weights/ones_like/ConstConst]^dnn/head/metrics/accuracy/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
5dnn/head/metrics/accuracy/broadcast_weights/ones_likeFill;dnn/head/metrics/accuracy/broadcast_weights/ones_like/Shape;dnn/head/metrics/accuracy/broadcast_weights/ones_like/Const*
T0*

index_type0*'
_output_shapes
:���������
�
+dnn/head/metrics/accuracy/broadcast_weightsMuldnn/head/metrics/Const5dnn/head/metrics/accuracy/broadcast_weights/ones_like*
T0*'
_output_shapes
:���������
�
dnn/head/metrics/accuracy/MulMuldnn/head/metrics/ToFloat+dnn/head/metrics/accuracy/broadcast_weights*
T0*'
_output_shapes
:���������
p
dnn/head/metrics/accuracy/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
dnn/head/metrics/accuracy/SumSum+dnn/head/metrics/accuracy/broadcast_weightsdnn/head/metrics/accuracy/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
r
!dnn/head/metrics/accuracy/Const_1Const*
valueB"       *
dtype0*
_output_shapes
:
�
dnn/head/metrics/accuracy/Sum_1Sumdnn/head/metrics/accuracy/Mul!dnn/head/metrics/accuracy/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
#dnn/head/metrics/accuracy/AssignAdd	AssignAdddnn/head/metrics/accuracy/totaldnn/head/metrics/accuracy/Sum_1*
T0*2
_class(
&$loc:@dnn/head/metrics/accuracy/total*
_output_shapes
: *
use_locking( 
�
%dnn/head/metrics/accuracy/AssignAdd_1	AssignAdddnn/head/metrics/accuracy/countdnn/head/metrics/accuracy/Sum^dnn/head/metrics/accuracy/Mul*2
_class(
&$loc:@dnn/head/metrics/accuracy/count*
_output_shapes
: *
use_locking( *
T0
�
!dnn/head/metrics/accuracy/truedivRealDiv$dnn/head/metrics/accuracy/total/read$dnn/head/metrics/accuracy/count/read*
_output_shapes
: *
T0
i
$dnn/head/metrics/accuracy/zeros_likeConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
!dnn/head/metrics/accuracy/GreaterGreater$dnn/head/metrics/accuracy/count/read$dnn/head/metrics/accuracy/zeros_like*
_output_shapes
: *
T0
�
dnn/head/metrics/accuracy/valueSelect!dnn/head/metrics/accuracy/Greater!dnn/head/metrics/accuracy/truediv$dnn/head/metrics/accuracy/zeros_like*
T0*
_output_shapes
: 
�
#dnn/head/metrics/accuracy/truediv_1RealDiv#dnn/head/metrics/accuracy/AssignAdd%dnn/head/metrics/accuracy/AssignAdd_1*
T0*
_output_shapes
: 
k
&dnn/head/metrics/accuracy/zeros_like_1Const*
valueB
 *    *
dtype0*
_output_shapes
: 
�
#dnn/head/metrics/accuracy/Greater_1Greater%dnn/head/metrics/accuracy/AssignAdd_1&dnn/head/metrics/accuracy/zeros_like_1*
T0*
_output_shapes
: 
�
#dnn/head/metrics/accuracy/update_opSelect#dnn/head/metrics/accuracy/Greater_1#dnn/head/metrics/accuracy/truediv_1&dnn/head/metrics/accuracy/zeros_like_1*
T0*
_output_shapes
: 
�
dnn/head/metrics/precision/CastCastdnn/head/predictions/ExpandDims*

SrcT0	*'
_output_shapes
:���������*

DstT0

�
!dnn/head/metrics/precision/Cast_1Castdnn/head/assert_range/Identity*

SrcT0*'
_output_shapes
:���������*

DstT0

e
 dnn/head/metrics/precision/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
s
1dnn/head/metrics/precision/true_positives/Equal/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 
�
/dnn/head/metrics/precision/true_positives/EqualEqual!dnn/head/metrics/precision/Cast_11dnn/head/metrics/precision/true_positives/Equal/y*
T0
*'
_output_shapes
:���������
u
3dnn/head/metrics/precision/true_positives/Equal_1/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 
�
1dnn/head/metrics/precision/true_positives/Equal_1Equaldnn/head/metrics/precision/Cast3dnn/head/metrics/precision/true_positives/Equal_1/y*'
_output_shapes
:���������*
T0

�
4dnn/head/metrics/precision/true_positives/LogicalAnd
LogicalAnd/dnn/head/metrics/precision/true_positives/Equal1dnn/head/metrics/precision/true_positives/Equal_1*'
_output_shapes
:���������
`
Xdnn/head/metrics/precision/true_positives/assert_type/statically_determined_correct_typeNoOp
�
Adnn/head/metrics/precision/true_positives/count/Initializer/zerosConst*B
_class8
64loc:@dnn/head/metrics/precision/true_positives/count*
valueB
 *    *
dtype0*
_output_shapes
: 
�
/dnn/head/metrics/precision/true_positives/count
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *B
_class8
64loc:@dnn/head/metrics/precision/true_positives/count
�
6dnn/head/metrics/precision/true_positives/count/AssignAssign/dnn/head/metrics/precision/true_positives/countAdnn/head/metrics/precision/true_positives/count/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@dnn/head/metrics/precision/true_positives/count*
validate_shape(*
_output_shapes
: 
�
4dnn/head/metrics/precision/true_positives/count/readIdentity/dnn/head/metrics/precision/true_positives/count*
T0*B
_class8
64loc:@dnn/head/metrics/precision/true_positives/count*
_output_shapes
: 
�
1dnn/head/metrics/precision/true_positives/ToFloatCast4dnn/head/metrics/precision/true_positives/LogicalAnd*

SrcT0
*'
_output_shapes
:���������*

DstT0
p
.dnn/head/metrics/precision/true_positives/RankConst*
value	B :*
dtype0*
_output_shapes
: 

=dnn/head/metrics/precision/true_positives/assert_rank_in/rankConst*
value	B : *
dtype0*
_output_shapes
: 
�
>dnn/head/metrics/precision/true_positives/assert_rank_in/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
o
gdnn/head/metrics/precision/true_positives/assert_rank_in/assert_type/statically_determined_correct_typeNoOp
q
idnn/head/metrics/precision/true_positives/assert_rank_in/assert_type_1/statically_determined_correct_typeNoOp
`
Xdnn/head/metrics/precision/true_positives/assert_rank_in/static_checks_determined_all_okNoOp
�
-dnn/head/metrics/precision/true_positives/MulMul1dnn/head/metrics/precision/true_positives/ToFloat dnn/head/metrics/precision/ConstY^dnn/head/metrics/precision/true_positives/assert_rank_in/static_checks_determined_all_ok*
T0*'
_output_shapes
:���������
�
2dnn/head/metrics/precision/true_positives/IdentityIdentity4dnn/head/metrics/precision/true_positives/count/read*
_output_shapes
: *
T0
�
/dnn/head/metrics/precision/true_positives/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
-dnn/head/metrics/precision/true_positives/SumSum-dnn/head/metrics/precision/true_positives/Mul/dnn/head/metrics/precision/true_positives/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
3dnn/head/metrics/precision/true_positives/AssignAdd	AssignAdd/dnn/head/metrics/precision/true_positives/count-dnn/head/metrics/precision/true_positives/Sum*B
_class8
64loc:@dnn/head/metrics/precision/true_positives/count*
_output_shapes
: *
use_locking( *
T0
t
2dnn/head/metrics/precision/false_positives/Equal/yConst*
value	B
 Z *
dtype0
*
_output_shapes
: 
�
0dnn/head/metrics/precision/false_positives/EqualEqual!dnn/head/metrics/precision/Cast_12dnn/head/metrics/precision/false_positives/Equal/y*
T0
*'
_output_shapes
:���������
v
4dnn/head/metrics/precision/false_positives/Equal_1/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 
�
2dnn/head/metrics/precision/false_positives/Equal_1Equaldnn/head/metrics/precision/Cast4dnn/head/metrics/precision/false_positives/Equal_1/y*'
_output_shapes
:���������*
T0

�
5dnn/head/metrics/precision/false_positives/LogicalAnd
LogicalAnd0dnn/head/metrics/precision/false_positives/Equal2dnn/head/metrics/precision/false_positives/Equal_1*'
_output_shapes
:���������
a
Ydnn/head/metrics/precision/false_positives/assert_type/statically_determined_correct_typeNoOp
�
Bdnn/head/metrics/precision/false_positives/count/Initializer/zerosConst*
dtype0*
_output_shapes
: *C
_class9
75loc:@dnn/head/metrics/precision/false_positives/count*
valueB
 *    
�
0dnn/head/metrics/precision/false_positives/count
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *C
_class9
75loc:@dnn/head/metrics/precision/false_positives/count
�
7dnn/head/metrics/precision/false_positives/count/AssignAssign0dnn/head/metrics/precision/false_positives/countBdnn/head/metrics/precision/false_positives/count/Initializer/zeros*
use_locking(*
T0*C
_class9
75loc:@dnn/head/metrics/precision/false_positives/count*
validate_shape(*
_output_shapes
: 
�
5dnn/head/metrics/precision/false_positives/count/readIdentity0dnn/head/metrics/precision/false_positives/count*
T0*C
_class9
75loc:@dnn/head/metrics/precision/false_positives/count*
_output_shapes
: 
�
2dnn/head/metrics/precision/false_positives/ToFloatCast5dnn/head/metrics/precision/false_positives/LogicalAnd*

SrcT0
*'
_output_shapes
:���������*

DstT0
q
/dnn/head/metrics/precision/false_positives/RankConst*
value	B :*
dtype0*
_output_shapes
: 
�
>dnn/head/metrics/precision/false_positives/assert_rank_in/rankConst*
value	B : *
dtype0*
_output_shapes
: 
�
?dnn/head/metrics/precision/false_positives/assert_rank_in/ShapeConst*
_output_shapes
: *
valueB *
dtype0
p
hdnn/head/metrics/precision/false_positives/assert_rank_in/assert_type/statically_determined_correct_typeNoOp
r
jdnn/head/metrics/precision/false_positives/assert_rank_in/assert_type_1/statically_determined_correct_typeNoOp
a
Ydnn/head/metrics/precision/false_positives/assert_rank_in/static_checks_determined_all_okNoOp
�
.dnn/head/metrics/precision/false_positives/MulMul2dnn/head/metrics/precision/false_positives/ToFloat dnn/head/metrics/precision/ConstZ^dnn/head/metrics/precision/false_positives/assert_rank_in/static_checks_determined_all_ok*
T0*'
_output_shapes
:���������
�
3dnn/head/metrics/precision/false_positives/IdentityIdentity5dnn/head/metrics/precision/false_positives/count/read*
T0*
_output_shapes
: 
�
0dnn/head/metrics/precision/false_positives/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
.dnn/head/metrics/precision/false_positives/SumSum.dnn/head/metrics/precision/false_positives/Mul0dnn/head/metrics/precision/false_positives/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
4dnn/head/metrics/precision/false_positives/AssignAdd	AssignAdd0dnn/head/metrics/precision/false_positives/count.dnn/head/metrics/precision/false_positives/Sum*
use_locking( *
T0*C
_class9
75loc:@dnn/head/metrics/precision/false_positives/count*
_output_shapes
: 
�
dnn/head/metrics/precision/addAdd2dnn/head/metrics/precision/true_positives/Identity3dnn/head/metrics/precision/false_positives/Identity*
T0*
_output_shapes
: 
i
$dnn/head/metrics/precision/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
"dnn/head/metrics/precision/GreaterGreaterdnn/head/metrics/precision/add$dnn/head/metrics/precision/Greater/y*
T0*
_output_shapes
: 
�
 dnn/head/metrics/precision/add_1Add2dnn/head/metrics/precision/true_positives/Identity3dnn/head/metrics/precision/false_positives/Identity*
T0*
_output_shapes
: 
�
dnn/head/metrics/precision/divRealDiv2dnn/head/metrics/precision/true_positives/Identity dnn/head/metrics/precision/add_1*
T0*
_output_shapes
: 
g
"dnn/head/metrics/precision/value/eConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
 dnn/head/metrics/precision/valueSelect"dnn/head/metrics/precision/Greaterdnn/head/metrics/precision/div"dnn/head/metrics/precision/value/e*
T0*
_output_shapes
: 
�
 dnn/head/metrics/precision/add_2Add3dnn/head/metrics/precision/true_positives/AssignAdd4dnn/head/metrics/precision/false_positives/AssignAdd*
T0*
_output_shapes
: 
k
&dnn/head/metrics/precision/Greater_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
$dnn/head/metrics/precision/Greater_1Greater dnn/head/metrics/precision/add_2&dnn/head/metrics/precision/Greater_1/y*
T0*
_output_shapes
: 
�
 dnn/head/metrics/precision/add_3Add3dnn/head/metrics/precision/true_positives/AssignAdd4dnn/head/metrics/precision/false_positives/AssignAdd*
T0*
_output_shapes
: 
�
 dnn/head/metrics/precision/div_1RealDiv3dnn/head/metrics/precision/true_positives/AssignAdd dnn/head/metrics/precision/add_3*
_output_shapes
: *
T0
k
&dnn/head/metrics/precision/update_op/eConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
$dnn/head/metrics/precision/update_opSelect$dnn/head/metrics/precision/Greater_1 dnn/head/metrics/precision/div_1&dnn/head/metrics/precision/update_op/e*
T0*
_output_shapes
: 
�
dnn/head/metrics/recall/CastCastdnn/head/predictions/ExpandDims*'
_output_shapes
:���������*

DstT0
*

SrcT0	
�
dnn/head/metrics/recall/Cast_1Castdnn/head/assert_range/Identity*'
_output_shapes
:���������*

DstT0
*

SrcT0
b
dnn/head/metrics/recall/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
p
.dnn/head/metrics/recall/true_positives/Equal/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 
�
,dnn/head/metrics/recall/true_positives/EqualEqualdnn/head/metrics/recall/Cast_1.dnn/head/metrics/recall/true_positives/Equal/y*
T0
*'
_output_shapes
:���������
r
0dnn/head/metrics/recall/true_positives/Equal_1/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 
�
.dnn/head/metrics/recall/true_positives/Equal_1Equaldnn/head/metrics/recall/Cast0dnn/head/metrics/recall/true_positives/Equal_1/y*
T0
*'
_output_shapes
:���������
�
1dnn/head/metrics/recall/true_positives/LogicalAnd
LogicalAnd,dnn/head/metrics/recall/true_positives/Equal.dnn/head/metrics/recall/true_positives/Equal_1*'
_output_shapes
:���������
]
Udnn/head/metrics/recall/true_positives/assert_type/statically_determined_correct_typeNoOp
�
>dnn/head/metrics/recall/true_positives/count/Initializer/zerosConst*?
_class5
31loc:@dnn/head/metrics/recall/true_positives/count*
valueB
 *    *
dtype0*
_output_shapes
: 
�
,dnn/head/metrics/recall/true_positives/count
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *?
_class5
31loc:@dnn/head/metrics/recall/true_positives/count*
	container 
�
3dnn/head/metrics/recall/true_positives/count/AssignAssign,dnn/head/metrics/recall/true_positives/count>dnn/head/metrics/recall/true_positives/count/Initializer/zeros*
_output_shapes
: *
use_locking(*
T0*?
_class5
31loc:@dnn/head/metrics/recall/true_positives/count*
validate_shape(
�
1dnn/head/metrics/recall/true_positives/count/readIdentity,dnn/head/metrics/recall/true_positives/count*
T0*?
_class5
31loc:@dnn/head/metrics/recall/true_positives/count*
_output_shapes
: 
�
.dnn/head/metrics/recall/true_positives/ToFloatCast1dnn/head/metrics/recall/true_positives/LogicalAnd*

SrcT0
*'
_output_shapes
:���������*

DstT0
m
+dnn/head/metrics/recall/true_positives/RankConst*
value	B :*
dtype0*
_output_shapes
: 
|
:dnn/head/metrics/recall/true_positives/assert_rank_in/rankConst*
_output_shapes
: *
value	B : *
dtype0
~
;dnn/head/metrics/recall/true_positives/assert_rank_in/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
l
ddnn/head/metrics/recall/true_positives/assert_rank_in/assert_type/statically_determined_correct_typeNoOp
n
fdnn/head/metrics/recall/true_positives/assert_rank_in/assert_type_1/statically_determined_correct_typeNoOp
]
Udnn/head/metrics/recall/true_positives/assert_rank_in/static_checks_determined_all_okNoOp
�
*dnn/head/metrics/recall/true_positives/MulMul.dnn/head/metrics/recall/true_positives/ToFloatdnn/head/metrics/recall/ConstV^dnn/head/metrics/recall/true_positives/assert_rank_in/static_checks_determined_all_ok*
T0*'
_output_shapes
:���������
�
/dnn/head/metrics/recall/true_positives/IdentityIdentity1dnn/head/metrics/recall/true_positives/count/read*
T0*
_output_shapes
: 
}
,dnn/head/metrics/recall/true_positives/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
*dnn/head/metrics/recall/true_positives/SumSum*dnn/head/metrics/recall/true_positives/Mul,dnn/head/metrics/recall/true_positives/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
0dnn/head/metrics/recall/true_positives/AssignAdd	AssignAdd,dnn/head/metrics/recall/true_positives/count*dnn/head/metrics/recall/true_positives/Sum*
use_locking( *
T0*?
_class5
31loc:@dnn/head/metrics/recall/true_positives/count*
_output_shapes
: 
q
/dnn/head/metrics/recall/false_negatives/Equal/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 
�
-dnn/head/metrics/recall/false_negatives/EqualEqualdnn/head/metrics/recall/Cast_1/dnn/head/metrics/recall/false_negatives/Equal/y*
T0
*'
_output_shapes
:���������
s
1dnn/head/metrics/recall/false_negatives/Equal_1/yConst*
value	B
 Z *
dtype0
*
_output_shapes
: 
�
/dnn/head/metrics/recall/false_negatives/Equal_1Equaldnn/head/metrics/recall/Cast1dnn/head/metrics/recall/false_negatives/Equal_1/y*'
_output_shapes
:���������*
T0

�
2dnn/head/metrics/recall/false_negatives/LogicalAnd
LogicalAnd-dnn/head/metrics/recall/false_negatives/Equal/dnn/head/metrics/recall/false_negatives/Equal_1*'
_output_shapes
:���������
^
Vdnn/head/metrics/recall/false_negatives/assert_type/statically_determined_correct_typeNoOp
�
?dnn/head/metrics/recall/false_negatives/count/Initializer/zerosConst*@
_class6
42loc:@dnn/head/metrics/recall/false_negatives/count*
valueB
 *    *
dtype0*
_output_shapes
: 
�
-dnn/head/metrics/recall/false_negatives/count
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *@
_class6
42loc:@dnn/head/metrics/recall/false_negatives/count*
	container 
�
4dnn/head/metrics/recall/false_negatives/count/AssignAssign-dnn/head/metrics/recall/false_negatives/count?dnn/head/metrics/recall/false_negatives/count/Initializer/zeros*
_output_shapes
: *
use_locking(*
T0*@
_class6
42loc:@dnn/head/metrics/recall/false_negatives/count*
validate_shape(
�
2dnn/head/metrics/recall/false_negatives/count/readIdentity-dnn/head/metrics/recall/false_negatives/count*
T0*@
_class6
42loc:@dnn/head/metrics/recall/false_negatives/count*
_output_shapes
: 
�
/dnn/head/metrics/recall/false_negatives/ToFloatCast2dnn/head/metrics/recall/false_negatives/LogicalAnd*'
_output_shapes
:���������*

DstT0*

SrcT0

n
,dnn/head/metrics/recall/false_negatives/RankConst*
value	B :*
dtype0*
_output_shapes
: 
}
;dnn/head/metrics/recall/false_negatives/assert_rank_in/rankConst*
value	B : *
dtype0*
_output_shapes
: 

<dnn/head/metrics/recall/false_negatives/assert_rank_in/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
m
ednn/head/metrics/recall/false_negatives/assert_rank_in/assert_type/statically_determined_correct_typeNoOp
o
gdnn/head/metrics/recall/false_negatives/assert_rank_in/assert_type_1/statically_determined_correct_typeNoOp
^
Vdnn/head/metrics/recall/false_negatives/assert_rank_in/static_checks_determined_all_okNoOp
�
+dnn/head/metrics/recall/false_negatives/MulMul/dnn/head/metrics/recall/false_negatives/ToFloatdnn/head/metrics/recall/ConstW^dnn/head/metrics/recall/false_negatives/assert_rank_in/static_checks_determined_all_ok*
T0*'
_output_shapes
:���������
�
0dnn/head/metrics/recall/false_negatives/IdentityIdentity2dnn/head/metrics/recall/false_negatives/count/read*
T0*
_output_shapes
: 
~
-dnn/head/metrics/recall/false_negatives/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
+dnn/head/metrics/recall/false_negatives/SumSum+dnn/head/metrics/recall/false_negatives/Mul-dnn/head/metrics/recall/false_negatives/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
1dnn/head/metrics/recall/false_negatives/AssignAdd	AssignAdd-dnn/head/metrics/recall/false_negatives/count+dnn/head/metrics/recall/false_negatives/Sum*
_output_shapes
: *
use_locking( *
T0*@
_class6
42loc:@dnn/head/metrics/recall/false_negatives/count
�
dnn/head/metrics/recall/addAdd/dnn/head/metrics/recall/true_positives/Identity0dnn/head/metrics/recall/false_negatives/Identity*
T0*
_output_shapes
: 
f
!dnn/head/metrics/recall/Greater/yConst*
_output_shapes
: *
valueB
 *    *
dtype0
�
dnn/head/metrics/recall/GreaterGreaterdnn/head/metrics/recall/add!dnn/head/metrics/recall/Greater/y*
_output_shapes
: *
T0
�
dnn/head/metrics/recall/add_1Add/dnn/head/metrics/recall/true_positives/Identity0dnn/head/metrics/recall/false_negatives/Identity*
_output_shapes
: *
T0
�
dnn/head/metrics/recall/divRealDiv/dnn/head/metrics/recall/true_positives/Identitydnn/head/metrics/recall/add_1*
T0*
_output_shapes
: 
d
dnn/head/metrics/recall/value/eConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
dnn/head/metrics/recall/valueSelectdnn/head/metrics/recall/Greaterdnn/head/metrics/recall/divdnn/head/metrics/recall/value/e*
T0*
_output_shapes
: 
�
dnn/head/metrics/recall/add_2Add0dnn/head/metrics/recall/true_positives/AssignAdd1dnn/head/metrics/recall/false_negatives/AssignAdd*
T0*
_output_shapes
: 
h
#dnn/head/metrics/recall/Greater_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
!dnn/head/metrics/recall/Greater_1Greaterdnn/head/metrics/recall/add_2#dnn/head/metrics/recall/Greater_1/y*
T0*
_output_shapes
: 
�
dnn/head/metrics/recall/add_3Add0dnn/head/metrics/recall/true_positives/AssignAdd1dnn/head/metrics/recall/false_negatives/AssignAdd*
T0*
_output_shapes
: 
�
dnn/head/metrics/recall/div_1RealDiv0dnn/head/metrics/recall/true_positives/AssignAdddnn/head/metrics/recall/add_3*
T0*
_output_shapes
: 
h
#dnn/head/metrics/recall/update_op/eConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
!dnn/head/metrics/recall/update_opSelect!dnn/head/metrics/recall/Greater_1dnn/head/metrics/recall/div_1#dnn/head/metrics/recall/update_op/e*
_output_shapes
: *
T0

:dnn/head/metrics/prediction/mean/broadcast_weights/weightsConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Udnn/head/metrics/prediction/mean/broadcast_weights/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Tdnn/head/metrics/prediction/mean/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
�
Tdnn/head/metrics/prediction/mean/broadcast_weights/assert_broadcastable/values/shapeShapednn/head/predictions/logistic*
_output_shapes
:*
T0*
out_type0
�
Sdnn/head/metrics/prediction/mean/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
k
cdnn/head/metrics/prediction/mean/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
�
Bdnn/head/metrics/prediction/mean/broadcast_weights/ones_like/ShapeShapednn/head/predictions/logisticd^dnn/head/metrics/prediction/mean/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
�
Bdnn/head/metrics/prediction/mean/broadcast_weights/ones_like/ConstConstd^dnn/head/metrics/prediction/mean/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
<dnn/head/metrics/prediction/mean/broadcast_weights/ones_likeFillBdnn/head/metrics/prediction/mean/broadcast_weights/ones_like/ShapeBdnn/head/metrics/prediction/mean/broadcast_weights/ones_like/Const*'
_output_shapes
:���������*
T0*

index_type0
�
2dnn/head/metrics/prediction/mean/broadcast_weightsMul:dnn/head/metrics/prediction/mean/broadcast_weights/weights<dnn/head/metrics/prediction/mean/broadcast_weights/ones_like*
T0*'
_output_shapes
:���������
�
8dnn/head/metrics/prediction/mean/total/Initializer/zerosConst*9
_class/
-+loc:@dnn/head/metrics/prediction/mean/total*
valueB
 *    *
dtype0*
_output_shapes
: 
�
&dnn/head/metrics/prediction/mean/total
VariableV2*
dtype0*
_output_shapes
: *
shared_name *9
_class/
-+loc:@dnn/head/metrics/prediction/mean/total*
	container *
shape: 
�
-dnn/head/metrics/prediction/mean/total/AssignAssign&dnn/head/metrics/prediction/mean/total8dnn/head/metrics/prediction/mean/total/Initializer/zeros*
use_locking(*
T0*9
_class/
-+loc:@dnn/head/metrics/prediction/mean/total*
validate_shape(*
_output_shapes
: 
�
+dnn/head/metrics/prediction/mean/total/readIdentity&dnn/head/metrics/prediction/mean/total*
T0*9
_class/
-+loc:@dnn/head/metrics/prediction/mean/total*
_output_shapes
: 
�
8dnn/head/metrics/prediction/mean/count/Initializer/zerosConst*9
_class/
-+loc:@dnn/head/metrics/prediction/mean/count*
valueB
 *    *
dtype0*
_output_shapes
: 
�
&dnn/head/metrics/prediction/mean/count
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *9
_class/
-+loc:@dnn/head/metrics/prediction/mean/count*
	container 
�
-dnn/head/metrics/prediction/mean/count/AssignAssign&dnn/head/metrics/prediction/mean/count8dnn/head/metrics/prediction/mean/count/Initializer/zeros*
use_locking(*
T0*9
_class/
-+loc:@dnn/head/metrics/prediction/mean/count*
validate_shape(*
_output_shapes
: 
�
+dnn/head/metrics/prediction/mean/count/readIdentity&dnn/head/metrics/prediction/mean/count*
T0*9
_class/
-+loc:@dnn/head/metrics/prediction/mean/count*
_output_shapes
: 
�
Wdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/shapeShape2dnn/head/metrics/prediction/mean/broadcast_weights*
T0*
out_type0*
_output_shapes
:
�
Vdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/rankConst*
value	B :*
dtype0*
_output_shapes
: 
�
Vdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/shapeShapednn/head/predictions/logistic*
_output_shapes
:*
T0*
out_type0
�
Udnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
�
Udnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalar/xConst*
_output_shapes
: *
value	B : *
dtype0
�
Sdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalarEqualUdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalar/xVdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/rank*
T0*
_output_shapes
: 
�
_dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/SwitchSwitchSdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalarSdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalar*
_output_shapes
: : *
T0

�
adnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_tIdentityadnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch:1*
T0
*
_output_shapes
: 
�
adnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_fIdentity_dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch*
_output_shapes
: *
T0

�
`dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_idIdentitySdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalar*
_output_shapes
: *
T0

�
adnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1SwitchSdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalar`dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0
*f
_class\
ZXloc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalar*
_output_shapes
: : 
�
dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqual�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1*
_output_shapes
: *
T0
�
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchSwitchUdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/rank`dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*h
_class^
\Zloc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/rank*
_output_shapes
: : 
�
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1SwitchVdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/rank`dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*i
_class_
][loc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/rank*
_output_shapes
: : 
�
ydnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/SwitchSwitchdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: : 
�
{dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_tIdentity{dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1*
_output_shapes
: *
T0

�
{dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_fIdentityydnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch*
_output_shapes
: *
T0

�
zdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_idIdentitydnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: 
�
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConst|^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
���������*
dtype0*
_output_shapes
: 
�
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDims�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim*

Tdim0*
T0*
_output_shapes

:
�
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchSwitchVdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/shape`dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*i
_class_
][loc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/shape* 
_output_shapes
::
�
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1Switch�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switchzdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*i
_class_
][loc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/shape* 
_output_shapes
::
�
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeConst|^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
_output_shapes
:*
valueB"      *
dtype0
�
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConst|^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
�
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFill�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const*
T0*

index_type0*
_output_shapes

:
�
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConst|^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
�
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis*
_output_shapes

:*

Tidx0*
T0*
N
�
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConst|^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
_output_shapes
: *
valueB :
���������*
dtype0
�
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDims�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes

:
�
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchSwitchWdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/shape`dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*j
_class`
^\loc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/shape* 
_output_shapes
::*
T0
�
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1Switch�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switchzdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id* 
_output_shapes
::*
T0*j
_class`
^\loc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/shape
�
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperation�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat*
validate_indices(*
T0*<
_output_shapes*
(:���������:���������:*
set_operationa-b
�
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSize�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1*
_output_shapes
: *
T0*
out_type0
�
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConst|^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B : *
dtype0*
_output_shapes
: 
�
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqual�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims*
T0*
_output_shapes
: 
�
{dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Switchdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankzdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0
*�
_class�
��loc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: : 
�
xdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeMerge{dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims*
T0
*
N*
_output_shapes
: : 
�
^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/MergeMergexdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Mergecdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1:1*
_output_shapes
: : *
T0
*
N
�
Odnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/ConstConst*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
�
Qdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/Const_1Const*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
�
Qdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/Const_2Const*
_output_shapes
: *E
value<B: B4dnn/head/metrics/prediction/mean/broadcast_weights:0*
dtype0
�
Qdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/Const_3Const*
_output_shapes
: *
valueB Bvalues.shape=*
dtype0
�
Qdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/Const_4Const*0
value'B% Bdnn/head/predictions/logistic:0*
dtype0*
_output_shapes
: 
�
Qdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/Const_5Const*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
�
\dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/SwitchSwitch^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: : 
�
^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_tIdentity^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Switch:1*
_output_shapes
: *
T0

�
^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_fIdentity\dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Switch*
_output_shapes
: *
T0

�
]dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_idIdentity^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: 
�
Zdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/NoOpNoOp_^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t
�
hdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependencyIdentity^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t[^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/NoOp*
T0
*q
_classg
ecloc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t*
_output_shapes
: 
�
cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0Const_^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
�
cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1Const_^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
�
cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2Const_^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: *E
value<B: B4dnn/head/metrics/prediction/mean/broadcast_weights:0*
dtype0
�
cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4Const_^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
�
cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5Const_^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
dtype0*
_output_shapes
: *0
value'B% Bdnn/head/predictions/logistic:0
�
cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7Const_^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
dtype0*
_output_shapes
: *
valueB B
is_scalar=
�	
\dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/AssertAssertcdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switchcdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2ednn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5ednn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7ednn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3*
T
2	
*
	summarize
�
cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/SwitchSwitch^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge]dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
_output_shapes
: : *
T0
*q
_classg
ecloc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge
�
ednn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1SwitchWdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/shape]dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0*j
_class`
^\loc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/shape* 
_output_shapes
::
�
ednn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2SwitchVdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/shape]dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id* 
_output_shapes
::*
T0*i
_class_
][loc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/shape
�
ednn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3SwitchSdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalar]dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
_output_shapes
: : *
T0
*f
_class\
ZXloc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalar
�
jdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency_1Identity^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f]^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert*
T0
*q
_classg
ecloc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: 
�
[dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/MergeMergejdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency_1hdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency*
N*
_output_shapes
: : *
T0

�
Ddnn/head/metrics/prediction/mean/broadcast_weights_1/ones_like/ShapeShapednn/head/predictions/logistic\^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Merge*
T0*
out_type0*
_output_shapes
:
�
Ddnn/head/metrics/prediction/mean/broadcast_weights_1/ones_like/ConstConst\^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Merge*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
>dnn/head/metrics/prediction/mean/broadcast_weights_1/ones_likeFillDdnn/head/metrics/prediction/mean/broadcast_weights_1/ones_like/ShapeDdnn/head/metrics/prediction/mean/broadcast_weights_1/ones_like/Const*'
_output_shapes
:���������*
T0*

index_type0
�
4dnn/head/metrics/prediction/mean/broadcast_weights_1Mul2dnn/head/metrics/prediction/mean/broadcast_weights>dnn/head/metrics/prediction/mean/broadcast_weights_1/ones_like*
T0*'
_output_shapes
:���������
�
$dnn/head/metrics/prediction/mean/MulMuldnn/head/predictions/logistic4dnn/head/metrics/prediction/mean/broadcast_weights_1*
T0*'
_output_shapes
:���������
w
&dnn/head/metrics/prediction/mean/ConstConst*
_output_shapes
:*
valueB"       *
dtype0
�
$dnn/head/metrics/prediction/mean/SumSum4dnn/head/metrics/prediction/mean/broadcast_weights_1&dnn/head/metrics/prediction/mean/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
y
(dnn/head/metrics/prediction/mean/Const_1Const*
_output_shapes
:*
valueB"       *
dtype0
�
&dnn/head/metrics/prediction/mean/Sum_1Sum$dnn/head/metrics/prediction/mean/Mul(dnn/head/metrics/prediction/mean/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
*dnn/head/metrics/prediction/mean/AssignAdd	AssignAdd&dnn/head/metrics/prediction/mean/total&dnn/head/metrics/prediction/mean/Sum_1*
_output_shapes
: *
use_locking( *
T0*9
_class/
-+loc:@dnn/head/metrics/prediction/mean/total
�
,dnn/head/metrics/prediction/mean/AssignAdd_1	AssignAdd&dnn/head/metrics/prediction/mean/count$dnn/head/metrics/prediction/mean/Sum%^dnn/head/metrics/prediction/mean/Mul*
_output_shapes
: *
use_locking( *
T0*9
_class/
-+loc:@dnn/head/metrics/prediction/mean/count
�
(dnn/head/metrics/prediction/mean/truedivRealDiv+dnn/head/metrics/prediction/mean/total/read+dnn/head/metrics/prediction/mean/count/read*
T0*
_output_shapes
: 
p
+dnn/head/metrics/prediction/mean/zeros_likeConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
(dnn/head/metrics/prediction/mean/GreaterGreater+dnn/head/metrics/prediction/mean/count/read+dnn/head/metrics/prediction/mean/zeros_like*
T0*
_output_shapes
: 
�
&dnn/head/metrics/prediction/mean/valueSelect(dnn/head/metrics/prediction/mean/Greater(dnn/head/metrics/prediction/mean/truediv+dnn/head/metrics/prediction/mean/zeros_like*
_output_shapes
: *
T0
�
*dnn/head/metrics/prediction/mean/truediv_1RealDiv*dnn/head/metrics/prediction/mean/AssignAdd,dnn/head/metrics/prediction/mean/AssignAdd_1*
_output_shapes
: *
T0
r
-dnn/head/metrics/prediction/mean/zeros_like_1Const*
valueB
 *    *
dtype0*
_output_shapes
: 
�
*dnn/head/metrics/prediction/mean/Greater_1Greater,dnn/head/metrics/prediction/mean/AssignAdd_1-dnn/head/metrics/prediction/mean/zeros_like_1*
_output_shapes
: *
T0
�
*dnn/head/metrics/prediction/mean/update_opSelect*dnn/head/metrics/prediction/mean/Greater_1*dnn/head/metrics/prediction/mean/truediv_1-dnn/head/metrics/prediction/mean/zeros_like_1*
_output_shapes
: *
T0
m
(dnn/head/metrics/accuracy_baseline/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
&dnn/head/metrics/accuracy_baseline/subSub(dnn/head/metrics/accuracy_baseline/sub/x!dnn/head/metrics/label/mean/value*
T0*
_output_shapes
: 
�
(dnn/head/metrics/accuracy_baseline/valueMaximum!dnn/head/metrics/label/mean/value&dnn/head/metrics/accuracy_baseline/sub*
_output_shapes
: *
T0
o
*dnn/head/metrics/accuracy_baseline/sub_1/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
(dnn/head/metrics/accuracy_baseline/sub_1Sub*dnn/head/metrics/accuracy_baseline/sub_1/x%dnn/head/metrics/label/mean/update_op*
T0*
_output_shapes
: 
�
,dnn/head/metrics/accuracy_baseline/update_opMaximum%dnn/head/metrics/label/mean/update_op(dnn/head/metrics/accuracy_baseline/sub_1*
_output_shapes
: *
T0
s
.dnn/head/metrics/auc/broadcast_weights/weightsConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
Idnn/head/metrics/auc/broadcast_weights/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Hdnn/head/metrics/auc/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
�
Hdnn/head/metrics/auc/broadcast_weights/assert_broadcastable/values/shapeShapednn/head/predictions/logistic*
T0*
out_type0*
_output_shapes
:
�
Gdnn/head/metrics/auc/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
_
Wdnn/head/metrics/auc/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
�
6dnn/head/metrics/auc/broadcast_weights/ones_like/ShapeShapednn/head/predictions/logisticX^dnn/head/metrics/auc/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
�
6dnn/head/metrics/auc/broadcast_weights/ones_like/ConstConstX^dnn/head/metrics/auc/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
0dnn/head/metrics/auc/broadcast_weights/ones_likeFill6dnn/head/metrics/auc/broadcast_weights/ones_like/Shape6dnn/head/metrics/auc/broadcast_weights/ones_like/Const*'
_output_shapes
:���������*
T0*

index_type0
�
&dnn/head/metrics/auc/broadcast_weightsMul.dnn/head/metrics/auc/broadcast_weights/weights0dnn/head/metrics/auc/broadcast_weights/ones_like*
T0*'
_output_shapes
:���������
`
dnn/head/metrics/auc/Cast/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
6dnn/head/metrics/auc/assert_greater_equal/GreaterEqualGreaterEqualdnn/head/predictions/logisticdnn/head/metrics/auc/Cast/x*
T0*'
_output_shapes
:���������
�
/dnn/head/metrics/auc/assert_greater_equal/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
-dnn/head/metrics/auc/assert_greater_equal/AllAll6dnn/head/metrics/auc/assert_greater_equal/GreaterEqual/dnn/head/metrics/auc/assert_greater_equal/Const*
_output_shapes
: *
	keep_dims( *

Tidx0
�
6dnn/head/metrics/auc/assert_greater_equal/Assert/ConstConst*.
value%B# Bpredictions must be in [0, 1]*
dtype0*
_output_shapes
: 
�
8dnn/head/metrics/auc/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *b
valueYBW BQCondition x >= y did not hold element-wise:x (dnn/head/predictions/logistic:0) = *
dtype0
�
8dnn/head/metrics/auc/assert_greater_equal/Assert/Const_2Const*5
value,B* B$y (dnn/head/metrics/auc/Cast/x:0) = *
dtype0*
_output_shapes
: 
�
Cdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/SwitchSwitch-dnn/head/metrics/auc/assert_greater_equal/All-dnn/head/metrics/auc/assert_greater_equal/All*
T0
*
_output_shapes
: : 
�
Ednn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_tIdentityEdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Switch:1*
_output_shapes
: *
T0

�
Ednn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_fIdentityCdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Switch*
T0
*
_output_shapes
: 
�
Ddnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/pred_idIdentity-dnn/head/metrics/auc/assert_greater_equal/All*
_output_shapes
: *
T0

�
Adnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/NoOpNoOpF^dnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_t
�
Odnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/control_dependencyIdentityEdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_tB^dnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*X
_classN
LJloc:@dnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_t*
_output_shapes
: 
�
Jdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/data_0ConstF^dnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_f*
_output_shapes
: *.
value%B# Bpredictions must be in [0, 1]*
dtype0
�
Jdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/data_1ConstF^dnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_f*b
valueYBW BQCondition x >= y did not hold element-wise:x (dnn/head/predictions/logistic:0) = *
dtype0*
_output_shapes
: 
�
Jdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/data_3ConstF^dnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_f*5
value,B* B$y (dnn/head/metrics/auc/Cast/x:0) = *
dtype0*
_output_shapes
: 
�
Cdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/AssertAssertJdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/SwitchJdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/data_0Jdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/data_1Ldnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/Switch_1Jdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/data_3Ldnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/Switch_2*
	summarize*
T	
2
�
Jdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/SwitchSwitch-dnn/head/metrics/auc/assert_greater_equal/AllDdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/pred_id*
T0
*@
_class6
42loc:@dnn/head/metrics/auc/assert_greater_equal/All*
_output_shapes
: : 
�
Ldnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/Switch_1Switchdnn/head/predictions/logisticDdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/pred_id*
T0*0
_class&
$"loc:@dnn/head/predictions/logistic*:
_output_shapes(
&:���������:���������
�
Ldnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/Switch_2Switchdnn/head/metrics/auc/Cast/xDdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/pred_id*
_output_shapes
: : *
T0*.
_class$
" loc:@dnn/head/metrics/auc/Cast/x
�
Qdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/control_dependency_1IdentityEdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_fD^dnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert*
T0
*X
_classN
LJloc:@dnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_f*
_output_shapes
: 
�
Bdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/MergeMergeQdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/control_dependency_1Odnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/control_dependency*
T0
*
N*
_output_shapes
: : 
b
dnn/head/metrics/auc/Cast_1/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
0dnn/head/metrics/auc/assert_less_equal/LessEqual	LessEqualdnn/head/predictions/logisticdnn/head/metrics/auc/Cast_1/x*'
_output_shapes
:���������*
T0
}
,dnn/head/metrics/auc/assert_less_equal/ConstConst*
_output_shapes
:*
valueB"       *
dtype0
�
*dnn/head/metrics/auc/assert_less_equal/AllAll0dnn/head/metrics/auc/assert_less_equal/LessEqual,dnn/head/metrics/auc/assert_less_equal/Const*
	keep_dims( *

Tidx0*
_output_shapes
: 
�
3dnn/head/metrics/auc/assert_less_equal/Assert/ConstConst*
_output_shapes
: *.
value%B# Bpredictions must be in [0, 1]*
dtype0
�
5dnn/head/metrics/auc/assert_less_equal/Assert/Const_1Const*b
valueYBW BQCondition x <= y did not hold element-wise:x (dnn/head/predictions/logistic:0) = *
dtype0*
_output_shapes
: 
�
5dnn/head/metrics/auc/assert_less_equal/Assert/Const_2Const*
dtype0*
_output_shapes
: *7
value.B, B&y (dnn/head/metrics/auc/Cast_1/x:0) = 
�
@dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/SwitchSwitch*dnn/head/metrics/auc/assert_less_equal/All*dnn/head/metrics/auc/assert_less_equal/All*
_output_shapes
: : *
T0

�
Bdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_tIdentityBdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Switch:1*
_output_shapes
: *
T0

�
Bdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_fIdentity@dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Switch*
T0
*
_output_shapes
: 
�
Adnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/pred_idIdentity*dnn/head/metrics/auc/assert_less_equal/All*
T0
*
_output_shapes
: 
�
>dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/NoOpNoOpC^dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_t
�
Ldnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/control_dependencyIdentityBdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_t?^dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/NoOp*
T0
*U
_classK
IGloc:@dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_t*
_output_shapes
: 
�
Gdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/data_0ConstC^dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_f*.
value%B# Bpredictions must be in [0, 1]*
dtype0*
_output_shapes
: 
�
Gdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/data_1ConstC^dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_f*b
valueYBW BQCondition x <= y did not hold element-wise:x (dnn/head/predictions/logistic:0) = *
dtype0*
_output_shapes
: 
�
Gdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/data_3ConstC^dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_f*
dtype0*
_output_shapes
: *7
value.B, B&y (dnn/head/metrics/auc/Cast_1/x:0) = 
�
@dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/AssertAssertGdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/SwitchGdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/data_0Gdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/data_1Idnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/Switch_1Gdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/data_3Idnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/Switch_2*
T	
2*
	summarize
�
Gdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/SwitchSwitch*dnn/head/metrics/auc/assert_less_equal/AllAdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/pred_id*=
_class3
1/loc:@dnn/head/metrics/auc/assert_less_equal/All*
_output_shapes
: : *
T0

�
Idnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/Switch_1Switchdnn/head/predictions/logisticAdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/pred_id*:
_output_shapes(
&:���������:���������*
T0*0
_class&
$"loc:@dnn/head/predictions/logistic
�
Idnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/Switch_2Switchdnn/head/metrics/auc/Cast_1/xAdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/pred_id*
_output_shapes
: : *
T0*0
_class&
$"loc:@dnn/head/metrics/auc/Cast_1/x
�
Ndnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/control_dependency_1IdentityBdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_fA^dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert*
T0
*U
_classK
IGloc:@dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_f*
_output_shapes
: 
�
?dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/MergeMergeNdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/control_dependency_1Ldnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/control_dependency*
T0
*
N*
_output_shapes
: : 
�
dnn/head/metrics/auc/Cast_2Castdnn/head/assert_range/IdentityC^dnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Merge@^dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Merge*'
_output_shapes
:���������*

DstT0
*

SrcT0
s
"dnn/head/metrics/auc/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"����   
�
dnn/head/metrics/auc/ReshapeReshapednn/head/predictions/logistic"dnn/head/metrics/auc/Reshape/shape*'
_output_shapes
:���������*
T0*
Tshape0
u
$dnn/head/metrics/auc/Reshape_1/shapeConst*
valueB"   ����*
dtype0*
_output_shapes
:
�
dnn/head/metrics/auc/Reshape_1Reshapednn/head/metrics/auc/Cast_2$dnn/head/metrics/auc/Reshape_1/shape*
T0
*
Tshape0*'
_output_shapes
:���������
v
dnn/head/metrics/auc/ShapeShapednn/head/metrics/auc/Reshape*
T0*
out_type0*
_output_shapes
:
r
(dnn/head/metrics/auc/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
t
*dnn/head/metrics/auc/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
t
*dnn/head/metrics/auc/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
"dnn/head/metrics/auc/strided_sliceStridedSlicednn/head/metrics/auc/Shape(dnn/head/metrics/auc/strided_slice/stack*dnn/head/metrics/auc/strided_slice/stack_1*dnn/head/metrics/auc/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
�
dnn/head/metrics/auc/ConstConst*�
value�B��"���ֳϩ�;ϩ$<��v<ϩ�<C��<���<�=ϩ$=	?9=C�M=}ib=��v=�Ʌ=��=2_�=ϩ�=l��=	?�=���=C��=��=}i�=��=���=�� >��>G�
>�>�9>2_>��>ϩ$>�)>l�.>�4>	?9>Wd>>��C>��H>C�M>��R>�X>.D]>}ib>ˎg>�l>h�q>��v>$|>���>Q7�>�Ʌ>�\�>G�>>��><��>�9�>�̗>2_�>��>���>(�>ϩ�>v<�>ϩ>�a�>l��>��>��>b��>	?�>�ѻ>Wd�>���>���>M�>���>�A�>C��>�f�>���>9��>��>���>.D�>���>}i�>$��>ˎ�>r!�>��>�F�>h��>l�>���>^��>$�>���>�� ?��?Q7?��?��?L?�\?�	?G�
?�8?�?B�?�?�]?<�?��?�9?7�?��?�?2_?��?��?-;?��?�� ?("?{`#?ϩ$?#�%?v<'?ʅ(?�)?q+?�a,?�-?l�.?�=0?�1?g�2?�4?c5?b�6?��7?	?9?]�:?��;?=?Wd>?��??��@?R@B?��C?��D?MF?�eG?��H?H�I?�AK?�L?C�M?�O?�fP?>�Q?��R?�BT?9�U?��V?�X?3hY?��Z?��[?.D]?��^?��_?) a?}ib?вc?$�d?xEf?ˎg?�h?r!j?�jk?�l?m�m?�Fo?�p?h�q?�"s?lt?c�u?��v?
Hx?^�y?��z?$|?Ym}?��~? �?*
dtype0*
_output_shapes	
:�
m
#dnn/head/metrics/auc/ExpandDims/dimConst*
valueB:*
dtype0*
_output_shapes
:
�
dnn/head/metrics/auc/ExpandDims
ExpandDimsdnn/head/metrics/auc/Const#dnn/head/metrics/auc/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:	�
^
dnn/head/metrics/auc/stack/0Const*
_output_shapes
: *
value	B :*
dtype0
�
dnn/head/metrics/auc/stackPackdnn/head/metrics/auc/stack/0"dnn/head/metrics/auc/strided_slice*
T0*

axis *
N*
_output_shapes
:
�
dnn/head/metrics/auc/TileTilednn/head/metrics/auc/ExpandDimsdnn/head/metrics/auc/stack*

Tmultiples0*
T0*(
_output_shapes
:����������
j
#dnn/head/metrics/auc/transpose/RankRankdnn/head/metrics/auc/Reshape*
T0*
_output_shapes
: 
f
$dnn/head/metrics/auc/transpose/sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
"dnn/head/metrics/auc/transpose/subSub#dnn/head/metrics/auc/transpose/Rank$dnn/head/metrics/auc/transpose/sub/y*
_output_shapes
: *
T0
l
*dnn/head/metrics/auc/transpose/Range/startConst*
dtype0*
_output_shapes
: *
value	B : 
l
*dnn/head/metrics/auc/transpose/Range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
�
$dnn/head/metrics/auc/transpose/RangeRange*dnn/head/metrics/auc/transpose/Range/start#dnn/head/metrics/auc/transpose/Rank*dnn/head/metrics/auc/transpose/Range/delta*
_output_shapes
:*

Tidx0
�
$dnn/head/metrics/auc/transpose/sub_1Sub"dnn/head/metrics/auc/transpose/sub$dnn/head/metrics/auc/transpose/Range*
_output_shapes
:*
T0
�
dnn/head/metrics/auc/transpose	Transposednn/head/metrics/auc/Reshape$dnn/head/metrics/auc/transpose/sub_1*
Tperm0*
T0*'
_output_shapes
:���������
v
%dnn/head/metrics/auc/Tile_1/multiplesConst*
_output_shapes
:*
valueB"�      *
dtype0
�
dnn/head/metrics/auc/Tile_1Tilednn/head/metrics/auc/transpose%dnn/head/metrics/auc/Tile_1/multiples*

Tmultiples0*
T0*(
_output_shapes
:����������
�
dnn/head/metrics/auc/GreaterGreaterdnn/head/metrics/auc/Tile_1dnn/head/metrics/auc/Tile*
T0*(
_output_shapes
:����������
u
dnn/head/metrics/auc/LogicalNot
LogicalNotdnn/head/metrics/auc/Greater*(
_output_shapes
:����������
v
%dnn/head/metrics/auc/Tile_2/multiplesConst*
_output_shapes
:*
valueB"�      *
dtype0
�
dnn/head/metrics/auc/Tile_2Tilednn/head/metrics/auc/Reshape_1%dnn/head/metrics/auc/Tile_2/multiples*
T0
*(
_output_shapes
:����������*

Tmultiples0
v
!dnn/head/metrics/auc/LogicalNot_1
LogicalNotdnn/head/metrics/auc/Tile_2*(
_output_shapes
:����������
�
Kdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/shapeShape&dnn/head/metrics/auc/broadcast_weights*
T0*
out_type0*
_output_shapes
:
�
Jdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/rankConst*
value	B :*
dtype0*
_output_shapes
: 
�
Jdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/shapeShapednn/head/predictions/logistic*
T0*
out_type0*
_output_shapes
:
�
Idnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
�
Idnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalar/xConst*
value	B : *
dtype0*
_output_shapes
: 
�
Gdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalarEqualIdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalar/xJdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/rank*
T0*
_output_shapes
: 
�
Sdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/SwitchSwitchGdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalarGdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
�
Udnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_tIdentityUdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch:1*
T0
*
_output_shapes
: 
�
Udnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_fIdentitySdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch*
T0
*
_output_shapes
: 
�
Tdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_idIdentityGdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: 
�
Udnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1SwitchGdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalarTdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0
*Z
_classP
NLloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalar*
_output_shapes
: : 
�
sdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqualzdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch|dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1*
_output_shapes
: *
T0
�
zdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchSwitchIdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/rankTdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*\
_classR
PNloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/rank*
_output_shapes
: : 
�
|dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1SwitchJdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/rankTdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*]
_classS
QOloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/rank*
_output_shapes
: : 
�
mdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/SwitchSwitchsdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_ranksdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: : *
T0

�
odnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_tIdentityodnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1*
T0
*
_output_shapes
: 
�
odnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_fIdentitymdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch*
T0
*
_output_shapes
: 
�
ndnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_idIdentitysdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: 
�
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConstp^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
���������*
dtype0*
_output_shapes
: 
�
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDims�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim*
T0*
_output_shapes

:*

Tdim0
�
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchSwitchJdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/shapeTdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*]
_classS
QOloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/shape* 
_output_shapes
::
�
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1Switch�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switchndnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id* 
_output_shapes
::*
T0*]
_classS
QOloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/shape
�
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeConstp^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB"      *
dtype0*
_output_shapes
:
�
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConstp^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
�
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFill�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const*

index_type0*
_output_shapes

:*
T0
�
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConstp^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
�
~dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis*

Tidx0*
T0*
N*
_output_shapes

:
�
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConstp^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
���������*
dtype0*
_output_shapes
: 
�
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDims�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim*
_output_shapes

:*

Tdim0*
T0
�
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchSwitchKdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/shapeTdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*^
_classT
RPloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/shape* 
_output_shapes
::
�
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1Switch�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switchndnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*^
_classT
RPloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/shape* 
_output_shapes
::
�
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperation�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1~dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat*<
_output_shapes*
(:���������:���������:*
set_operationa-b*
validate_indices(*
T0
�
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSize�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1*
_output_shapes
: *
T0*
out_type0
�
ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConstp^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
_output_shapes
: *
value	B : 
�
wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqualydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims*
T0*
_output_shapes
: 
�
odnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Switchsdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankndnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
_output_shapes
: : *
T0
*�
_class|
zxloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank
�
ldnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeMergeodnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims*
T0
*
N*
_output_shapes
: : 
�
Rdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/MergeMergeldnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeWdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1:1*
T0
*
N*
_output_shapes
: : 
�
Cdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/ConstConst*
dtype0*
_output_shapes
: *8
value/B- B'weights can not be broadcast to values.
�
Ednn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/Const_1Const*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
�
Ednn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/Const_2Const*9
value0B. B(dnn/head/metrics/auc/broadcast_weights:0*
dtype0*
_output_shapes
: 
�
Ednn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/Const_3Const*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
�
Ednn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/Const_4Const*0
value'B% Bdnn/head/predictions/logistic:0*
dtype0*
_output_shapes
: 
�
Ednn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/Const_5Const*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
�
Pdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/SwitchSwitchRdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/MergeRdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: : *
T0

�
Rdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_tIdentityRdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Switch:1*
_output_shapes
: *
T0

�
Rdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_fIdentityPdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Switch*
T0
*
_output_shapes
: 
�
Qdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_idIdentityRdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: *
T0

�
Ndnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/NoOpNoOpS^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t
�
\dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependencyIdentityRdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_tO^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/NoOp*
_output_shapes
: *
T0
*e
_class[
YWloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t
�
Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0ConstS^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
�
Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1ConstS^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: *
valueB Bweights.shape=*
dtype0
�
Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2ConstS^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*9
value0B. B(dnn/head/metrics/auc/broadcast_weights:0*
dtype0*
_output_shapes
: 
�
Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4ConstS^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
�
Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5ConstS^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*0
value'B% Bdnn/head/predictions/logistic:0*
dtype0*
_output_shapes
: 
�
Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7ConstS^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
�
Pdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/AssertAssertWdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/SwitchWdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2Ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5Ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7Ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3*
T
2	
*
	summarize
�
Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/SwitchSwitchRdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/MergeQdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0
*e
_class[
YWloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: : 
�
Ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1SwitchKdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/shapeQdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0*^
_classT
RPloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/shape* 
_output_shapes
::
�
Ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2SwitchJdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/shapeQdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id* 
_output_shapes
::*
T0*]
_classS
QOloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/shape
�
Ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3SwitchGdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalarQdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0
*Z
_classP
NLloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalar*
_output_shapes
: : 
�
^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency_1IdentityRdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_fQ^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert*e
_class[
YWloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: *
T0

�
Odnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/MergeMerge^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency_1\dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency*
T0
*
N*
_output_shapes
: : 
�
8dnn/head/metrics/auc/broadcast_weights_1/ones_like/ShapeShapednn/head/predictions/logisticP^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Merge*
T0*
out_type0*
_output_shapes
:
�
8dnn/head/metrics/auc/broadcast_weights_1/ones_like/ConstConstP^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Merge*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
2dnn/head/metrics/auc/broadcast_weights_1/ones_likeFill8dnn/head/metrics/auc/broadcast_weights_1/ones_like/Shape8dnn/head/metrics/auc/broadcast_weights_1/ones_like/Const*
T0*

index_type0*'
_output_shapes
:���������
�
(dnn/head/metrics/auc/broadcast_weights_1Mul&dnn/head/metrics/auc/broadcast_weights2dnn/head/metrics/auc/broadcast_weights_1/ones_like*'
_output_shapes
:���������*
T0
u
$dnn/head/metrics/auc/Reshape_2/shapeConst*
valueB"   ����*
dtype0*
_output_shapes
:
�
dnn/head/metrics/auc/Reshape_2Reshape(dnn/head/metrics/auc/broadcast_weights_1$dnn/head/metrics/auc/Reshape_2/shape*'
_output_shapes
:���������*
T0*
Tshape0
v
%dnn/head/metrics/auc/Tile_3/multiplesConst*
valueB"�      *
dtype0*
_output_shapes
:
�
dnn/head/metrics/auc/Tile_3Tilednn/head/metrics/auc/Reshape_2%dnn/head/metrics/auc/Tile_3/multiples*(
_output_shapes
:����������*

Tmultiples0*
T0
�
5dnn/head/metrics/auc/true_positives/Initializer/zerosConst*6
_class,
*(loc:@dnn/head/metrics/auc/true_positives*
valueB�*    *
dtype0*
_output_shapes	
:�
�
#dnn/head/metrics/auc/true_positives
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *6
_class,
*(loc:@dnn/head/metrics/auc/true_positives*
	container *
shape:�
�
*dnn/head/metrics/auc/true_positives/AssignAssign#dnn/head/metrics/auc/true_positives5dnn/head/metrics/auc/true_positives/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*6
_class,
*(loc:@dnn/head/metrics/auc/true_positives
�
(dnn/head/metrics/auc/true_positives/readIdentity#dnn/head/metrics/auc/true_positives*
T0*6
_class,
*(loc:@dnn/head/metrics/auc/true_positives*
_output_shapes	
:�
�
dnn/head/metrics/auc/LogicalAnd
LogicalAnddnn/head/metrics/auc/Tile_2dnn/head/metrics/auc/Greater*(
_output_shapes
:����������
�
dnn/head/metrics/auc/ToFloat_2Castdnn/head/metrics/auc/LogicalAnd*

SrcT0
*(
_output_shapes
:����������*

DstT0
�
dnn/head/metrics/auc/mulMuldnn/head/metrics/auc/ToFloat_2dnn/head/metrics/auc/Tile_3*
T0*(
_output_shapes
:����������
l
*dnn/head/metrics/auc/Sum/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B :
�
dnn/head/metrics/auc/SumSumdnn/head/metrics/auc/mul*dnn/head/metrics/auc/Sum/reduction_indices*
T0*
_output_shapes	
:�*
	keep_dims( *

Tidx0
�
dnn/head/metrics/auc/AssignAdd	AssignAdd#dnn/head/metrics/auc/true_positivesdnn/head/metrics/auc/Sum*
_output_shapes	
:�*
use_locking( *
T0*6
_class,
*(loc:@dnn/head/metrics/auc/true_positives
�
6dnn/head/metrics/auc/false_negatives/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*7
_class-
+)loc:@dnn/head/metrics/auc/false_negatives*
valueB�*    
�
$dnn/head/metrics/auc/false_negatives
VariableV2*
shared_name *7
_class-
+)loc:@dnn/head/metrics/auc/false_negatives*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
+dnn/head/metrics/auc/false_negatives/AssignAssign$dnn/head/metrics/auc/false_negatives6dnn/head/metrics/auc/false_negatives/Initializer/zeros*
T0*7
_class-
+)loc:@dnn/head/metrics/auc/false_negatives*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
)dnn/head/metrics/auc/false_negatives/readIdentity$dnn/head/metrics/auc/false_negatives*
T0*7
_class-
+)loc:@dnn/head/metrics/auc/false_negatives*
_output_shapes	
:�
�
!dnn/head/metrics/auc/LogicalAnd_1
LogicalAnddnn/head/metrics/auc/Tile_2dnn/head/metrics/auc/LogicalNot*(
_output_shapes
:����������
�
dnn/head/metrics/auc/ToFloat_3Cast!dnn/head/metrics/auc/LogicalAnd_1*

SrcT0
*(
_output_shapes
:����������*

DstT0
�
dnn/head/metrics/auc/mul_1Muldnn/head/metrics/auc/ToFloat_3dnn/head/metrics/auc/Tile_3*(
_output_shapes
:����������*
T0
n
,dnn/head/metrics/auc/Sum_1/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
�
dnn/head/metrics/auc/Sum_1Sumdnn/head/metrics/auc/mul_1,dnn/head/metrics/auc/Sum_1/reduction_indices*
	keep_dims( *

Tidx0*
T0*
_output_shapes	
:�
�
 dnn/head/metrics/auc/AssignAdd_1	AssignAdd$dnn/head/metrics/auc/false_negativesdnn/head/metrics/auc/Sum_1*
_output_shapes	
:�*
use_locking( *
T0*7
_class-
+)loc:@dnn/head/metrics/auc/false_negatives
�
5dnn/head/metrics/auc/true_negatives/Initializer/zerosConst*
_output_shapes	
:�*6
_class,
*(loc:@dnn/head/metrics/auc/true_negatives*
valueB�*    *
dtype0
�
#dnn/head/metrics/auc/true_negatives
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *6
_class,
*(loc:@dnn/head/metrics/auc/true_negatives
�
*dnn/head/metrics/auc/true_negatives/AssignAssign#dnn/head/metrics/auc/true_negatives5dnn/head/metrics/auc/true_negatives/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@dnn/head/metrics/auc/true_negatives*
validate_shape(*
_output_shapes	
:�
�
(dnn/head/metrics/auc/true_negatives/readIdentity#dnn/head/metrics/auc/true_negatives*
T0*6
_class,
*(loc:@dnn/head/metrics/auc/true_negatives*
_output_shapes	
:�
�
!dnn/head/metrics/auc/LogicalAnd_2
LogicalAnd!dnn/head/metrics/auc/LogicalNot_1dnn/head/metrics/auc/LogicalNot*(
_output_shapes
:����������
�
dnn/head/metrics/auc/ToFloat_4Cast!dnn/head/metrics/auc/LogicalAnd_2*

SrcT0
*(
_output_shapes
:����������*

DstT0
�
dnn/head/metrics/auc/mul_2Muldnn/head/metrics/auc/ToFloat_4dnn/head/metrics/auc/Tile_3*
T0*(
_output_shapes
:����������
n
,dnn/head/metrics/auc/Sum_2/reduction_indicesConst*
_output_shapes
: *
value	B :*
dtype0
�
dnn/head/metrics/auc/Sum_2Sumdnn/head/metrics/auc/mul_2,dnn/head/metrics/auc/Sum_2/reduction_indices*
T0*
_output_shapes	
:�*
	keep_dims( *

Tidx0
�
 dnn/head/metrics/auc/AssignAdd_2	AssignAdd#dnn/head/metrics/auc/true_negativesdnn/head/metrics/auc/Sum_2*
_output_shapes	
:�*
use_locking( *
T0*6
_class,
*(loc:@dnn/head/metrics/auc/true_negatives
�
6dnn/head/metrics/auc/false_positives/Initializer/zerosConst*
_output_shapes	
:�*7
_class-
+)loc:@dnn/head/metrics/auc/false_positives*
valueB�*    *
dtype0
�
$dnn/head/metrics/auc/false_positives
VariableV2*
_output_shapes	
:�*
shared_name *7
_class-
+)loc:@dnn/head/metrics/auc/false_positives*
	container *
shape:�*
dtype0
�
+dnn/head/metrics/auc/false_positives/AssignAssign$dnn/head/metrics/auc/false_positives6dnn/head/metrics/auc/false_positives/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*7
_class-
+)loc:@dnn/head/metrics/auc/false_positives*
validate_shape(
�
)dnn/head/metrics/auc/false_positives/readIdentity$dnn/head/metrics/auc/false_positives*
T0*7
_class-
+)loc:@dnn/head/metrics/auc/false_positives*
_output_shapes	
:�
�
!dnn/head/metrics/auc/LogicalAnd_3
LogicalAnd!dnn/head/metrics/auc/LogicalNot_1dnn/head/metrics/auc/Greater*(
_output_shapes
:����������
�
dnn/head/metrics/auc/ToFloat_5Cast!dnn/head/metrics/auc/LogicalAnd_3*(
_output_shapes
:����������*

DstT0*

SrcT0

�
dnn/head/metrics/auc/mul_3Muldnn/head/metrics/auc/ToFloat_5dnn/head/metrics/auc/Tile_3*(
_output_shapes
:����������*
T0
n
,dnn/head/metrics/auc/Sum_3/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
�
dnn/head/metrics/auc/Sum_3Sumdnn/head/metrics/auc/mul_3,dnn/head/metrics/auc/Sum_3/reduction_indices*
T0*
_output_shapes	
:�*
	keep_dims( *

Tidx0
�
 dnn/head/metrics/auc/AssignAdd_3	AssignAdd$dnn/head/metrics/auc/false_positivesdnn/head/metrics/auc/Sum_3*
use_locking( *
T0*7
_class-
+)loc:@dnn/head/metrics/auc/false_positives*
_output_shapes	
:�
_
dnn/head/metrics/auc/add/yConst*
_output_shapes
: *
valueB
 *�7�5*
dtype0
�
dnn/head/metrics/auc/addAdd(dnn/head/metrics/auc/true_positives/readdnn/head/metrics/auc/add/y*
_output_shapes	
:�*
T0
�
dnn/head/metrics/auc/add_1Add(dnn/head/metrics/auc/true_positives/read)dnn/head/metrics/auc/false_negatives/read*
_output_shapes	
:�*
T0
a
dnn/head/metrics/auc/add_2/yConst*
_output_shapes
: *
valueB
 *�7�5*
dtype0
�
dnn/head/metrics/auc/add_2Adddnn/head/metrics/auc/add_1dnn/head/metrics/auc/add_2/y*
_output_shapes	
:�*
T0

dnn/head/metrics/auc/divRealDivdnn/head/metrics/auc/adddnn/head/metrics/auc/add_2*
T0*
_output_shapes	
:�
�
dnn/head/metrics/auc/add_3Add)dnn/head/metrics/auc/false_positives/read(dnn/head/metrics/auc/true_negatives/read*
_output_shapes	
:�*
T0
a
dnn/head/metrics/auc/add_4/yConst*
valueB
 *�7�5*
dtype0*
_output_shapes
: 
�
dnn/head/metrics/auc/add_4Adddnn/head/metrics/auc/add_3dnn/head/metrics/auc/add_4/y*
T0*
_output_shapes	
:�
�
dnn/head/metrics/auc/div_1RealDiv)dnn/head/metrics/auc/false_positives/readdnn/head/metrics/auc/add_4*
_output_shapes	
:�*
T0
t
*dnn/head/metrics/auc/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
w
,dnn/head/metrics/auc/strided_slice_1/stack_1Const*
valueB:�*
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_1/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
�
$dnn/head/metrics/auc/strided_slice_1StridedSlicednn/head/metrics/auc/div_1*dnn/head/metrics/auc/strided_slice_1/stack,dnn/head/metrics/auc/strided_slice_1/stack_1,dnn/head/metrics/auc/strided_slice_1/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes	
:�
t
*dnn/head/metrics/auc/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
$dnn/head/metrics/auc/strided_slice_2StridedSlicednn/head/metrics/auc/div_1*dnn/head/metrics/auc/strided_slice_2/stack,dnn/head/metrics/auc/strided_slice_2/stack_1,dnn/head/metrics/auc/strided_slice_2/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes	
:�*
T0*
Index0*
shrink_axis_mask 
�
dnn/head/metrics/auc/subSub$dnn/head/metrics/auc/strided_slice_1$dnn/head/metrics/auc/strided_slice_2*
T0*
_output_shapes	
:�
t
*dnn/head/metrics/auc/strided_slice_3/stackConst*
_output_shapes
:*
valueB: *
dtype0
w
,dnn/head/metrics/auc/strided_slice_3/stack_1Const*
valueB:�*
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
$dnn/head/metrics/auc/strided_slice_3StridedSlicednn/head/metrics/auc/div*dnn/head/metrics/auc/strided_slice_3/stack,dnn/head/metrics/auc/strided_slice_3/stack_1,dnn/head/metrics/auc/strided_slice_3/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask *
_output_shapes	
:�*
Index0*
T0
t
*dnn/head/metrics/auc/strided_slice_4/stackConst*
valueB:*
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_4/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_4/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
$dnn/head/metrics/auc/strided_slice_4StridedSlicednn/head/metrics/auc/div*dnn/head/metrics/auc/strided_slice_4/stack,dnn/head/metrics/auc/strided_slice_4/stack_1,dnn/head/metrics/auc/strided_slice_4/stack_2*
end_mask*
_output_shapes	
:�*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask 
�
dnn/head/metrics/auc/add_5Add$dnn/head/metrics/auc/strided_slice_3$dnn/head/metrics/auc/strided_slice_4*
T0*
_output_shapes	
:�
c
dnn/head/metrics/auc/truediv/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
dnn/head/metrics/auc/truedivRealDivdnn/head/metrics/auc/add_5dnn/head/metrics/auc/truediv/y*
T0*
_output_shapes	
:�
}
dnn/head/metrics/auc/MulMuldnn/head/metrics/auc/subdnn/head/metrics/auc/truediv*
T0*
_output_shapes	
:�
f
dnn/head/metrics/auc/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
dnn/head/metrics/auc/valueSumdnn/head/metrics/auc/Muldnn/head/metrics/auc/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
a
dnn/head/metrics/auc/add_6/yConst*
valueB
 *�7�5*
dtype0*
_output_shapes
: 
�
dnn/head/metrics/auc/add_6Adddnn/head/metrics/auc/AssignAdddnn/head/metrics/auc/add_6/y*
T0*
_output_shapes	
:�
�
dnn/head/metrics/auc/add_7Adddnn/head/metrics/auc/AssignAdd dnn/head/metrics/auc/AssignAdd_1*
_output_shapes	
:�*
T0
a
dnn/head/metrics/auc/add_8/yConst*
valueB
 *�7�5*
dtype0*
_output_shapes
: 
�
dnn/head/metrics/auc/add_8Adddnn/head/metrics/auc/add_7dnn/head/metrics/auc/add_8/y*
_output_shapes	
:�*
T0
�
dnn/head/metrics/auc/div_2RealDivdnn/head/metrics/auc/add_6dnn/head/metrics/auc/add_8*
T0*
_output_shapes	
:�
�
dnn/head/metrics/auc/add_9Add dnn/head/metrics/auc/AssignAdd_3 dnn/head/metrics/auc/AssignAdd_2*
_output_shapes	
:�*
T0
b
dnn/head/metrics/auc/add_10/yConst*
valueB
 *�7�5*
dtype0*
_output_shapes
: 
�
dnn/head/metrics/auc/add_10Adddnn/head/metrics/auc/add_9dnn/head/metrics/auc/add_10/y*
T0*
_output_shapes	
:�
�
dnn/head/metrics/auc/div_3RealDiv dnn/head/metrics/auc/AssignAdd_3dnn/head/metrics/auc/add_10*
T0*
_output_shapes	
:�
t
*dnn/head/metrics/auc/strided_slice_5/stackConst*
valueB: *
dtype0*
_output_shapes
:
w
,dnn/head/metrics/auc/strided_slice_5/stack_1Const*
_output_shapes
:*
valueB:�*
dtype0
v
,dnn/head/metrics/auc/strided_slice_5/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
$dnn/head/metrics/auc/strided_slice_5StridedSlicednn/head/metrics/auc/div_3*dnn/head/metrics/auc/strided_slice_5/stack,dnn/head/metrics/auc/strided_slice_5/stack_1,dnn/head/metrics/auc/strided_slice_5/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes	
:�
t
*dnn/head/metrics/auc/strided_slice_6/stackConst*
valueB:*
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_6/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_6/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
$dnn/head/metrics/auc/strided_slice_6StridedSlicednn/head/metrics/auc/div_3*dnn/head/metrics/auc/strided_slice_6/stack,dnn/head/metrics/auc/strided_slice_6/stack_1,dnn/head/metrics/auc/strided_slice_6/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes	
:�*
T0*
Index0*
shrink_axis_mask 
�
dnn/head/metrics/auc/sub_1Sub$dnn/head/metrics/auc/strided_slice_5$dnn/head/metrics/auc/strided_slice_6*
T0*
_output_shapes	
:�
t
*dnn/head/metrics/auc/strided_slice_7/stackConst*
valueB: *
dtype0*
_output_shapes
:
w
,dnn/head/metrics/auc/strided_slice_7/stack_1Const*
valueB:�*
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_7/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
$dnn/head/metrics/auc/strided_slice_7StridedSlicednn/head/metrics/auc/div_2*dnn/head/metrics/auc/strided_slice_7/stack,dnn/head/metrics/auc/strided_slice_7/stack_1,dnn/head/metrics/auc/strided_slice_7/stack_2*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask *
_output_shapes	
:�*
T0*
Index0*
shrink_axis_mask 
t
*dnn/head/metrics/auc/strided_slice_8/stackConst*
valueB:*
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_8/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_8/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
$dnn/head/metrics/auc/strided_slice_8StridedSlicednn/head/metrics/auc/div_2*dnn/head/metrics/auc/strided_slice_8/stack,dnn/head/metrics/auc/strided_slice_8/stack_1,dnn/head/metrics/auc/strided_slice_8/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes	
:�*
T0*
Index0
�
dnn/head/metrics/auc/add_11Add$dnn/head/metrics/auc/strided_slice_7$dnn/head/metrics/auc/strided_slice_8*
T0*
_output_shapes	
:�
e
 dnn/head/metrics/auc/truediv_1/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
dnn/head/metrics/auc/truediv_1RealDivdnn/head/metrics/auc/add_11 dnn/head/metrics/auc/truediv_1/y*
_output_shapes	
:�*
T0
�
dnn/head/metrics/auc/Mul_1Muldnn/head/metrics/auc/sub_1dnn/head/metrics/auc/truediv_1*
_output_shapes	
:�*
T0
f
dnn/head/metrics/auc/Const_2Const*
valueB: *
dtype0*
_output_shapes
:
�
dnn/head/metrics/auc/update_opSumdnn/head/metrics/auc/Mul_1dnn/head/metrics/auc/Const_2*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
?dnn/head/metrics/auc_precision_recall/broadcast_weights/weightsConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Zdnn/head/metrics/auc_precision_recall/broadcast_weights/assert_broadcastable/weights/shapeConst*
dtype0*
_output_shapes
: *
valueB 
�
Ydnn/head/metrics/auc_precision_recall/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
�
Ydnn/head/metrics/auc_precision_recall/broadcast_weights/assert_broadcastable/values/shapeShapednn/head/predictions/logistic*
T0*
out_type0*
_output_shapes
:
�
Xdnn/head/metrics/auc_precision_recall/broadcast_weights/assert_broadcastable/values/rankConst*
_output_shapes
: *
value	B :*
dtype0
p
hdnn/head/metrics/auc_precision_recall/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
�
Gdnn/head/metrics/auc_precision_recall/broadcast_weights/ones_like/ShapeShapednn/head/predictions/logistici^dnn/head/metrics/auc_precision_recall/broadcast_weights/assert_broadcastable/static_scalar_check_success*
_output_shapes
:*
T0*
out_type0
�
Gdnn/head/metrics/auc_precision_recall/broadcast_weights/ones_like/ConstConsti^dnn/head/metrics/auc_precision_recall/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Adnn/head/metrics/auc_precision_recall/broadcast_weights/ones_likeFillGdnn/head/metrics/auc_precision_recall/broadcast_weights/ones_like/ShapeGdnn/head/metrics/auc_precision_recall/broadcast_weights/ones_like/Const*
T0*

index_type0*'
_output_shapes
:���������
�
7dnn/head/metrics/auc_precision_recall/broadcast_weightsMul?dnn/head/metrics/auc_precision_recall/broadcast_weights/weightsAdnn/head/metrics/auc_precision_recall/broadcast_weights/ones_like*
T0*'
_output_shapes
:���������
q
,dnn/head/metrics/auc_precision_recall/Cast/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Gdnn/head/metrics/auc_precision_recall/assert_greater_equal/GreaterEqualGreaterEqualdnn/head/predictions/logistic,dnn/head/metrics/auc_precision_recall/Cast/x*
T0*'
_output_shapes
:���������
�
@dnn/head/metrics/auc_precision_recall/assert_greater_equal/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
>dnn/head/metrics/auc_precision_recall/assert_greater_equal/AllAllGdnn/head/metrics/auc_precision_recall/assert_greater_equal/GreaterEqual@dnn/head/metrics/auc_precision_recall/assert_greater_equal/Const*
_output_shapes
: *
	keep_dims( *

Tidx0
�
Gdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/ConstConst*.
value%B# Bpredictions must be in [0, 1]*
dtype0*
_output_shapes
: 
�
Idnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/Const_1Const*b
valueYBW BQCondition x >= y did not hold element-wise:x (dnn/head/predictions/logistic:0) = *
dtype0*
_output_shapes
: 
�
Idnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/Const_2Const*F
value=B; B5y (dnn/head/metrics/auc_precision_recall/Cast/x:0) = *
dtype0*
_output_shapes
: 
�
Tdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/SwitchSwitch>dnn/head/metrics/auc_precision_recall/assert_greater_equal/All>dnn/head/metrics/auc_precision_recall/assert_greater_equal/All*
_output_shapes
: : *
T0

�
Vdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_tIdentityVdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
�
Vdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_fIdentityTdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Switch*
_output_shapes
: *
T0

�
Udnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/pred_idIdentity>dnn/head/metrics/auc_precision_recall/assert_greater_equal/All*
T0
*
_output_shapes
: 
�
Rdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/NoOpNoOpW^dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_t
�
`dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/control_dependencyIdentityVdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_tS^dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*i
_class_
][loc:@dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_t*
_output_shapes
: 
�
[dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/data_0ConstW^dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_f*.
value%B# Bpredictions must be in [0, 1]*
dtype0*
_output_shapes
: 
�
[dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/data_1ConstW^dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_f*b
valueYBW BQCondition x >= y did not hold element-wise:x (dnn/head/predictions/logistic:0) = *
dtype0*
_output_shapes
: 
�
[dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/data_3ConstW^dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_f*
_output_shapes
: *F
value=B; B5y (dnn/head/metrics/auc_precision_recall/Cast/x:0) = *
dtype0
�
Tdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/AssertAssert[dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/Switch[dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/data_0[dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/data_1]dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/Switch_1[dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/data_3]dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/Switch_2*
T	
2*
	summarize
�
[dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/SwitchSwitch>dnn/head/metrics/auc_precision_recall/assert_greater_equal/AllUdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/pred_id*
T0
*Q
_classG
ECloc:@dnn/head/metrics/auc_precision_recall/assert_greater_equal/All*
_output_shapes
: : 
�
]dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/Switch_1Switchdnn/head/predictions/logisticUdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/pred_id*
T0*0
_class&
$"loc:@dnn/head/predictions/logistic*:
_output_shapes(
&:���������:���������
�
]dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/Switch_2Switch,dnn/head/metrics/auc_precision_recall/Cast/xUdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/pred_id*
T0*?
_class5
31loc:@dnn/head/metrics/auc_precision_recall/Cast/x*
_output_shapes
: : 
�
bdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/control_dependency_1IdentityVdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_fU^dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert*
T0
*i
_class_
][loc:@dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_f*
_output_shapes
: 
�
Sdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/MergeMergebdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/control_dependency_1`dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/control_dependency*
T0
*
N*
_output_shapes
: : 
s
.dnn/head/metrics/auc_precision_recall/Cast_1/xConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
�
Adnn/head/metrics/auc_precision_recall/assert_less_equal/LessEqual	LessEqualdnn/head/predictions/logistic.dnn/head/metrics/auc_precision_recall/Cast_1/x*
T0*'
_output_shapes
:���������
�
=dnn/head/metrics/auc_precision_recall/assert_less_equal/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
;dnn/head/metrics/auc_precision_recall/assert_less_equal/AllAllAdnn/head/metrics/auc_precision_recall/assert_less_equal/LessEqual=dnn/head/metrics/auc_precision_recall/assert_less_equal/Const*
_output_shapes
: *
	keep_dims( *

Tidx0
�
Ddnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/ConstConst*.
value%B# Bpredictions must be in [0, 1]*
dtype0*
_output_shapes
: 
�
Fdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/Const_1Const*
dtype0*
_output_shapes
: *b
valueYBW BQCondition x <= y did not hold element-wise:x (dnn/head/predictions/logistic:0) = 
�
Fdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/Const_2Const*
_output_shapes
: *H
value?B= B7y (dnn/head/metrics/auc_precision_recall/Cast_1/x:0) = *
dtype0
�
Qdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/SwitchSwitch;dnn/head/metrics/auc_precision_recall/assert_less_equal/All;dnn/head/metrics/auc_precision_recall/assert_less_equal/All*
T0
*
_output_shapes
: : 
�
Sdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_tIdentitySdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
�
Sdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_fIdentityQdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Switch*
T0
*
_output_shapes
: 
�
Rdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/pred_idIdentity;dnn/head/metrics/auc_precision_recall/assert_less_equal/All*
T0
*
_output_shapes
: 
�
Odnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/NoOpNoOpT^dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_t
�
]dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/control_dependencyIdentitySdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_tP^dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/NoOp*
T0
*f
_class\
ZXloc:@dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_t*
_output_shapes
: 
�
Xdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/data_0ConstT^dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_f*.
value%B# Bpredictions must be in [0, 1]*
dtype0*
_output_shapes
: 
�
Xdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/data_1ConstT^dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_f*b
valueYBW BQCondition x <= y did not hold element-wise:x (dnn/head/predictions/logistic:0) = *
dtype0*
_output_shapes
: 
�
Xdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/data_3ConstT^dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_f*H
value?B= B7y (dnn/head/metrics/auc_precision_recall/Cast_1/x:0) = *
dtype0*
_output_shapes
: 
�
Qdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/AssertAssertXdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/SwitchXdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/data_0Xdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/data_1Zdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/Switch_1Xdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/data_3Zdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/Switch_2*
T	
2*
	summarize
�
Xdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/SwitchSwitch;dnn/head/metrics/auc_precision_recall/assert_less_equal/AllRdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/pred_id*
T0
*N
_classD
B@loc:@dnn/head/metrics/auc_precision_recall/assert_less_equal/All*
_output_shapes
: : 
�
Zdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/Switch_1Switchdnn/head/predictions/logisticRdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/pred_id*0
_class&
$"loc:@dnn/head/predictions/logistic*:
_output_shapes(
&:���������:���������*
T0
�
Zdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/Switch_2Switch.dnn/head/metrics/auc_precision_recall/Cast_1/xRdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/pred_id*
T0*A
_class7
53loc:@dnn/head/metrics/auc_precision_recall/Cast_1/x*
_output_shapes
: : 
�
_dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/control_dependency_1IdentitySdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_fR^dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert*
T0
*f
_class\
ZXloc:@dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_f*
_output_shapes
: 
�
Pdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/MergeMerge_dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/control_dependency_1]dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/control_dependency*
_output_shapes
: : *
T0
*
N
�
,dnn/head/metrics/auc_precision_recall/Cast_2Castdnn/head/assert_range/IdentityT^dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/MergeQ^dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Merge*

SrcT0*'
_output_shapes
:���������*

DstT0

�
3dnn/head/metrics/auc_precision_recall/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"����   
�
-dnn/head/metrics/auc_precision_recall/ReshapeReshapednn/head/predictions/logistic3dnn/head/metrics/auc_precision_recall/Reshape/shape*'
_output_shapes
:���������*
T0*
Tshape0
�
5dnn/head/metrics/auc_precision_recall/Reshape_1/shapeConst*
valueB"   ����*
dtype0*
_output_shapes
:
�
/dnn/head/metrics/auc_precision_recall/Reshape_1Reshape,dnn/head/metrics/auc_precision_recall/Cast_25dnn/head/metrics/auc_precision_recall/Reshape_1/shape*
T0
*
Tshape0*'
_output_shapes
:���������
�
+dnn/head/metrics/auc_precision_recall/ShapeShape-dnn/head/metrics/auc_precision_recall/Reshape*
T0*
out_type0*
_output_shapes
:
�
9dnn/head/metrics/auc_precision_recall/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
;dnn/head/metrics/auc_precision_recall/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
;dnn/head/metrics/auc_precision_recall/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
3dnn/head/metrics/auc_precision_recall/strided_sliceStridedSlice+dnn/head/metrics/auc_precision_recall/Shape9dnn/head/metrics/auc_precision_recall/strided_slice/stack;dnn/head/metrics/auc_precision_recall/strided_slice/stack_1;dnn/head/metrics/auc_precision_recall/strided_slice/stack_2*
_output_shapes
: *
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
�
+dnn/head/metrics/auc_precision_recall/ConstConst*�
value�B��"���ֳϩ�;ϩ$<��v<ϩ�<C��<���<�=ϩ$=	?9=C�M=}ib=��v=�Ʌ=��=2_�=ϩ�=l��=	?�=���=C��=��=}i�=��=���=�� >��>G�
>�>�9>2_>��>ϩ$>�)>l�.>�4>	?9>Wd>>��C>��H>C�M>��R>�X>.D]>}ib>ˎg>�l>h�q>��v>$|>���>Q7�>�Ʌ>�\�>G�>>��><��>�9�>�̗>2_�>��>���>(�>ϩ�>v<�>ϩ>�a�>l��>��>��>b��>	?�>�ѻ>Wd�>���>���>M�>���>�A�>C��>�f�>���>9��>��>���>.D�>���>}i�>$��>ˎ�>r!�>��>�F�>h��>l�>���>^��>$�>���>�� ?��?Q7?��?��?L?�\?�	?G�
?�8?�?B�?�?�]?<�?��?�9?7�?��?�?2_?��?��?-;?��?�� ?("?{`#?ϩ$?#�%?v<'?ʅ(?�)?q+?�a,?�-?l�.?�=0?�1?g�2?�4?c5?b�6?��7?	?9?]�:?��;?=?Wd>?��??��@?R@B?��C?��D?MF?�eG?��H?H�I?�AK?�L?C�M?�O?�fP?>�Q?��R?�BT?9�U?��V?�X?3hY?��Z?��[?.D]?��^?��_?) a?}ib?вc?$�d?xEf?ˎg?�h?r!j?�jk?�l?m�m?�Fo?�p?h�q?�"s?lt?c�u?��v?
Hx?^�y?��z?$|?Ym}?��~? �?*
dtype0*
_output_shapes	
:�
~
4dnn/head/metrics/auc_precision_recall/ExpandDims/dimConst*
valueB:*
dtype0*
_output_shapes
:
�
0dnn/head/metrics/auc_precision_recall/ExpandDims
ExpandDims+dnn/head/metrics/auc_precision_recall/Const4dnn/head/metrics/auc_precision_recall/ExpandDims/dim*
_output_shapes
:	�*

Tdim0*
T0
o
-dnn/head/metrics/auc_precision_recall/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 
�
+dnn/head/metrics/auc_precision_recall/stackPack-dnn/head/metrics/auc_precision_recall/stack/03dnn/head/metrics/auc_precision_recall/strided_slice*
T0*

axis *
N*
_output_shapes
:
�
*dnn/head/metrics/auc_precision_recall/TileTile0dnn/head/metrics/auc_precision_recall/ExpandDims+dnn/head/metrics/auc_precision_recall/stack*

Tmultiples0*
T0*(
_output_shapes
:����������
�
4dnn/head/metrics/auc_precision_recall/transpose/RankRank-dnn/head/metrics/auc_precision_recall/Reshape*
T0*
_output_shapes
: 
w
5dnn/head/metrics/auc_precision_recall/transpose/sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
3dnn/head/metrics/auc_precision_recall/transpose/subSub4dnn/head/metrics/auc_precision_recall/transpose/Rank5dnn/head/metrics/auc_precision_recall/transpose/sub/y*
T0*
_output_shapes
: 
}
;dnn/head/metrics/auc_precision_recall/transpose/Range/startConst*
_output_shapes
: *
value	B : *
dtype0
}
;dnn/head/metrics/auc_precision_recall/transpose/Range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
5dnn/head/metrics/auc_precision_recall/transpose/RangeRange;dnn/head/metrics/auc_precision_recall/transpose/Range/start4dnn/head/metrics/auc_precision_recall/transpose/Rank;dnn/head/metrics/auc_precision_recall/transpose/Range/delta*
_output_shapes
:*

Tidx0
�
5dnn/head/metrics/auc_precision_recall/transpose/sub_1Sub3dnn/head/metrics/auc_precision_recall/transpose/sub5dnn/head/metrics/auc_precision_recall/transpose/Range*
T0*
_output_shapes
:
�
/dnn/head/metrics/auc_precision_recall/transpose	Transpose-dnn/head/metrics/auc_precision_recall/Reshape5dnn/head/metrics/auc_precision_recall/transpose/sub_1*
T0*'
_output_shapes
:���������*
Tperm0
�
6dnn/head/metrics/auc_precision_recall/Tile_1/multiplesConst*
valueB"�      *
dtype0*
_output_shapes
:
�
,dnn/head/metrics/auc_precision_recall/Tile_1Tile/dnn/head/metrics/auc_precision_recall/transpose6dnn/head/metrics/auc_precision_recall/Tile_1/multiples*(
_output_shapes
:����������*

Tmultiples0*
T0
�
-dnn/head/metrics/auc_precision_recall/GreaterGreater,dnn/head/metrics/auc_precision_recall/Tile_1*dnn/head/metrics/auc_precision_recall/Tile*
T0*(
_output_shapes
:����������
�
0dnn/head/metrics/auc_precision_recall/LogicalNot
LogicalNot-dnn/head/metrics/auc_precision_recall/Greater*(
_output_shapes
:����������
�
6dnn/head/metrics/auc_precision_recall/Tile_2/multiplesConst*
dtype0*
_output_shapes
:*
valueB"�      
�
,dnn/head/metrics/auc_precision_recall/Tile_2Tile/dnn/head/metrics/auc_precision_recall/Reshape_16dnn/head/metrics/auc_precision_recall/Tile_2/multiples*

Tmultiples0*
T0
*(
_output_shapes
:����������
�
2dnn/head/metrics/auc_precision_recall/LogicalNot_1
LogicalNot,dnn/head/metrics/auc_precision_recall/Tile_2*(
_output_shapes
:����������
�
\dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/shapeShape7dnn/head/metrics/auc_precision_recall/broadcast_weights*
_output_shapes
:*
T0*
out_type0
�
[dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/rankConst*
value	B :*
dtype0*
_output_shapes
: 
�
[dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/shapeShapednn/head/predictions/logistic*
T0*
out_type0*
_output_shapes
:
�
Zdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
�
Zdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalar/xConst*
value	B : *
dtype0*
_output_shapes
: 
�
Xdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalarEqualZdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalar/x[dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/rank*
T0*
_output_shapes
: 
�
ddnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/SwitchSwitchXdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalarXdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
�
fdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_tIdentityfdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch:1*
_output_shapes
: *
T0

�
fdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_fIdentityddnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch*
_output_shapes
: *
T0

�
ednn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_idIdentityXdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalar*
_output_shapes
: *
T0

�
fdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1SwitchXdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalarednn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
_output_shapes
: : *
T0
*k
_classa
_]loc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalar
�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqual�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1*
T0*
_output_shapes
: 
�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchSwitchZdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/rankednn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
_output_shapes
: : *
T0*m
_classc
a_loc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/rank
�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1Switch[dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/rankednn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
_output_shapes
: : *
T0*n
_classd
b`loc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/rank
�
~dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/SwitchSwitch�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: : 
�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_tIdentity�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1*
_output_shapes
: *
T0

�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_fIdentity~dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch*
T0
*
_output_shapes
: 
�
dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_idIdentity�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: 
�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConst�^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
���������*
dtype0*
_output_shapes
: 
�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDims�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim*
T0*
_output_shapes

:*

Tdim0
�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchSwitch[dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/shapeednn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*n
_classd
b`loc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/shape* 
_output_shapes
::
�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1Switch�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switchdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*n
_classd
b`loc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/shape* 
_output_shapes
::
�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeConst�^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB"      *
dtype0*
_output_shapes
:
�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConst�^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
_output_shapes
: *
value	B :
�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFill�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const*
T0*

index_type0*
_output_shapes

:
�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConst�^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis*

Tidx0*
T0*
N*
_output_shapes

:
�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConst�^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
���������*
dtype0*
_output_shapes
: 
�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDims�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim*
_output_shapes

:*

Tdim0*
T0
�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchSwitch\dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/shapeednn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*o
_classe
caloc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/shape* 
_output_shapes
::
�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1Switch�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switchdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id* 
_output_shapes
::*
T0*o
_classe
caloc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/shape
�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperation�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat*
validate_indices(*
T0*<
_output_shapes*
(:���������:���������:*
set_operationa-b
�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSize�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1*
T0*
out_type0*
_output_shapes
: 
�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConst�^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B : *
dtype0*
_output_shapes
: 
�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqual�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims*
T0*
_output_shapes
: 
�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Switch�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0
*�
_class�
��loc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: : 
�
}dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeMerge�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims*
T0
*
N*
_output_shapes
: : 
�
cdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/MergeMerge}dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Mergehdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1:1*
T0
*
N*
_output_shapes
: : 
�
Tdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/ConstConst*
_output_shapes
: *8
value/B- B'weights can not be broadcast to values.*
dtype0
�
Vdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/Const_1Const*
_output_shapes
: *
valueB Bweights.shape=*
dtype0
�
Vdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/Const_2Const*J
valueAB? B9dnn/head/metrics/auc_precision_recall/broadcast_weights:0*
dtype0*
_output_shapes
: 
�
Vdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/Const_3Const*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
�
Vdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/Const_4Const*0
value'B% Bdnn/head/predictions/logistic:0*
dtype0*
_output_shapes
: 
�
Vdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/Const_5Const*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
�
adnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/SwitchSwitchcdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Mergecdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: : 
�
cdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_tIdentitycdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Switch:1*
_output_shapes
: *
T0

�
cdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_fIdentityadnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Switch*
_output_shapes
: *
T0

�
bdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_idIdentitycdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: *
T0

�
_dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/NoOpNoOpd^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t
�
mdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependencyIdentitycdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t`^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/NoOp*
T0
*v
_classl
jhloc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t*
_output_shapes
: 
�
hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0Constd^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
�
hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1Constd^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
dtype0*
_output_shapes
: *
valueB Bweights.shape=
�
hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2Constd^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*J
valueAB? B9dnn/head/metrics/auc_precision_recall/broadcast_weights:0*
dtype0*
_output_shapes
: 
�
hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4Constd^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
dtype0*
_output_shapes
: *
valueB Bvalues.shape=
�
hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5Constd^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*0
value'B% Bdnn/head/predictions/logistic:0*
dtype0*
_output_shapes
: 
�
hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7Constd^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
dtype0*
_output_shapes
: *
valueB B
is_scalar=
�	
adnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/AssertAsserthdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switchhdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2jdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5jdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7jdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3*
T
2	
*
	summarize
�
hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/SwitchSwitchcdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Mergebdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0
*v
_classl
jhloc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: : 
�
jdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1Switch\dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/shapebdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0*o
_classe
caloc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/shape* 
_output_shapes
::
�
jdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2Switch[dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/shapebdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*n
_classd
b`loc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/shape* 
_output_shapes
::*
T0
�
jdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3SwitchXdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalarbdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0
*k
_classa
_]loc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalar*
_output_shapes
: : 
�
odnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency_1Identitycdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_fb^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert*
T0
*v
_classl
jhloc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: 
�
`dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/MergeMergeodnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency_1mdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency*
_output_shapes
: : *
T0
*
N
�
Idnn/head/metrics/auc_precision_recall/broadcast_weights_1/ones_like/ShapeShapednn/head/predictions/logistica^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Merge*
_output_shapes
:*
T0*
out_type0
�
Idnn/head/metrics/auc_precision_recall/broadcast_weights_1/ones_like/ConstConsta^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Merge*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Cdnn/head/metrics/auc_precision_recall/broadcast_weights_1/ones_likeFillIdnn/head/metrics/auc_precision_recall/broadcast_weights_1/ones_like/ShapeIdnn/head/metrics/auc_precision_recall/broadcast_weights_1/ones_like/Const*'
_output_shapes
:���������*
T0*

index_type0
�
9dnn/head/metrics/auc_precision_recall/broadcast_weights_1Mul7dnn/head/metrics/auc_precision_recall/broadcast_weightsCdnn/head/metrics/auc_precision_recall/broadcast_weights_1/ones_like*
T0*'
_output_shapes
:���������
�
5dnn/head/metrics/auc_precision_recall/Reshape_2/shapeConst*
_output_shapes
:*
valueB"   ����*
dtype0
�
/dnn/head/metrics/auc_precision_recall/Reshape_2Reshape9dnn/head/metrics/auc_precision_recall/broadcast_weights_15dnn/head/metrics/auc_precision_recall/Reshape_2/shape*'
_output_shapes
:���������*
T0*
Tshape0
�
6dnn/head/metrics/auc_precision_recall/Tile_3/multiplesConst*
valueB"�      *
dtype0*
_output_shapes
:
�
,dnn/head/metrics/auc_precision_recall/Tile_3Tile/dnn/head/metrics/auc_precision_recall/Reshape_26dnn/head/metrics/auc_precision_recall/Tile_3/multiples*

Tmultiples0*
T0*(
_output_shapes
:����������
�
Fdnn/head/metrics/auc_precision_recall/true_positives/Initializer/zerosConst*G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_positives*
valueB�*    *
dtype0*
_output_shapes	
:�
�
4dnn/head/metrics/auc_precision_recall/true_positives
VariableV2*
shared_name *G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_positives*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
;dnn/head/metrics/auc_precision_recall/true_positives/AssignAssign4dnn/head/metrics/auc_precision_recall/true_positivesFdnn/head/metrics/auc_precision_recall/true_positives/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_positives*
validate_shape(*
_output_shapes	
:�
�
9dnn/head/metrics/auc_precision_recall/true_positives/readIdentity4dnn/head/metrics/auc_precision_recall/true_positives*
T0*G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_positives*
_output_shapes	
:�
�
0dnn/head/metrics/auc_precision_recall/LogicalAnd
LogicalAnd,dnn/head/metrics/auc_precision_recall/Tile_2-dnn/head/metrics/auc_precision_recall/Greater*(
_output_shapes
:����������
�
/dnn/head/metrics/auc_precision_recall/ToFloat_2Cast0dnn/head/metrics/auc_precision_recall/LogicalAnd*

SrcT0
*(
_output_shapes
:����������*

DstT0
�
)dnn/head/metrics/auc_precision_recall/mulMul/dnn/head/metrics/auc_precision_recall/ToFloat_2,dnn/head/metrics/auc_precision_recall/Tile_3*(
_output_shapes
:����������*
T0
}
;dnn/head/metrics/auc_precision_recall/Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
�
)dnn/head/metrics/auc_precision_recall/SumSum)dnn/head/metrics/auc_precision_recall/mul;dnn/head/metrics/auc_precision_recall/Sum/reduction_indices*
_output_shapes	
:�*
	keep_dims( *

Tidx0*
T0
�
/dnn/head/metrics/auc_precision_recall/AssignAdd	AssignAdd4dnn/head/metrics/auc_precision_recall/true_positives)dnn/head/metrics/auc_precision_recall/Sum*G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_positives*
_output_shapes	
:�*
use_locking( *
T0
�
Gdnn/head/metrics/auc_precision_recall/false_negatives/Initializer/zerosConst*H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_negatives*
valueB�*    *
dtype0*
_output_shapes	
:�
�
5dnn/head/metrics/auc_precision_recall/false_negatives
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_negatives
�
<dnn/head/metrics/auc_precision_recall/false_negatives/AssignAssign5dnn/head/metrics/auc_precision_recall/false_negativesGdnn/head/metrics/auc_precision_recall/false_negatives/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_negatives*
validate_shape(*
_output_shapes	
:�
�
:dnn/head/metrics/auc_precision_recall/false_negatives/readIdentity5dnn/head/metrics/auc_precision_recall/false_negatives*H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_negatives*
_output_shapes	
:�*
T0
�
2dnn/head/metrics/auc_precision_recall/LogicalAnd_1
LogicalAnd,dnn/head/metrics/auc_precision_recall/Tile_20dnn/head/metrics/auc_precision_recall/LogicalNot*(
_output_shapes
:����������
�
/dnn/head/metrics/auc_precision_recall/ToFloat_3Cast2dnn/head/metrics/auc_precision_recall/LogicalAnd_1*(
_output_shapes
:����������*

DstT0*

SrcT0

�
+dnn/head/metrics/auc_precision_recall/mul_1Mul/dnn/head/metrics/auc_precision_recall/ToFloat_3,dnn/head/metrics/auc_precision_recall/Tile_3*
T0*(
_output_shapes
:����������

=dnn/head/metrics/auc_precision_recall/Sum_1/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
�
+dnn/head/metrics/auc_precision_recall/Sum_1Sum+dnn/head/metrics/auc_precision_recall/mul_1=dnn/head/metrics/auc_precision_recall/Sum_1/reduction_indices*
	keep_dims( *

Tidx0*
T0*
_output_shapes	
:�
�
1dnn/head/metrics/auc_precision_recall/AssignAdd_1	AssignAdd5dnn/head/metrics/auc_precision_recall/false_negatives+dnn/head/metrics/auc_precision_recall/Sum_1*
_output_shapes	
:�*
use_locking( *
T0*H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_negatives
�
Fdnn/head/metrics/auc_precision_recall/true_negatives/Initializer/zerosConst*G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_negatives*
valueB�*    *
dtype0*
_output_shapes	
:�
�
4dnn/head/metrics/auc_precision_recall/true_negatives
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_negatives
�
;dnn/head/metrics/auc_precision_recall/true_negatives/AssignAssign4dnn/head/metrics/auc_precision_recall/true_negativesFdnn/head/metrics/auc_precision_recall/true_negatives/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_negatives*
validate_shape(*
_output_shapes	
:�
�
9dnn/head/metrics/auc_precision_recall/true_negatives/readIdentity4dnn/head/metrics/auc_precision_recall/true_negatives*
_output_shapes	
:�*
T0*G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_negatives
�
2dnn/head/metrics/auc_precision_recall/LogicalAnd_2
LogicalAnd2dnn/head/metrics/auc_precision_recall/LogicalNot_10dnn/head/metrics/auc_precision_recall/LogicalNot*(
_output_shapes
:����������
�
/dnn/head/metrics/auc_precision_recall/ToFloat_4Cast2dnn/head/metrics/auc_precision_recall/LogicalAnd_2*

SrcT0
*(
_output_shapes
:����������*

DstT0
�
+dnn/head/metrics/auc_precision_recall/mul_2Mul/dnn/head/metrics/auc_precision_recall/ToFloat_4,dnn/head/metrics/auc_precision_recall/Tile_3*
T0*(
_output_shapes
:����������

=dnn/head/metrics/auc_precision_recall/Sum_2/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
�
+dnn/head/metrics/auc_precision_recall/Sum_2Sum+dnn/head/metrics/auc_precision_recall/mul_2=dnn/head/metrics/auc_precision_recall/Sum_2/reduction_indices*
T0*
_output_shapes	
:�*
	keep_dims( *

Tidx0
�
1dnn/head/metrics/auc_precision_recall/AssignAdd_2	AssignAdd4dnn/head/metrics/auc_precision_recall/true_negatives+dnn/head/metrics/auc_precision_recall/Sum_2*G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_negatives*
_output_shapes	
:�*
use_locking( *
T0
�
Gdnn/head/metrics/auc_precision_recall/false_positives/Initializer/zerosConst*H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_positives*
valueB�*    *
dtype0*
_output_shapes	
:�
�
5dnn/head/metrics/auc_precision_recall/false_positives
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_positives*
	container *
shape:�
�
<dnn/head/metrics/auc_precision_recall/false_positives/AssignAssign5dnn/head/metrics/auc_precision_recall/false_positivesGdnn/head/metrics/auc_precision_recall/false_positives/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_positives*
validate_shape(*
_output_shapes	
:�
�
:dnn/head/metrics/auc_precision_recall/false_positives/readIdentity5dnn/head/metrics/auc_precision_recall/false_positives*
_output_shapes	
:�*
T0*H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_positives
�
2dnn/head/metrics/auc_precision_recall/LogicalAnd_3
LogicalAnd2dnn/head/metrics/auc_precision_recall/LogicalNot_1-dnn/head/metrics/auc_precision_recall/Greater*(
_output_shapes
:����������
�
/dnn/head/metrics/auc_precision_recall/ToFloat_5Cast2dnn/head/metrics/auc_precision_recall/LogicalAnd_3*(
_output_shapes
:����������*

DstT0*

SrcT0

�
+dnn/head/metrics/auc_precision_recall/mul_3Mul/dnn/head/metrics/auc_precision_recall/ToFloat_5,dnn/head/metrics/auc_precision_recall/Tile_3*
T0*(
_output_shapes
:����������

=dnn/head/metrics/auc_precision_recall/Sum_3/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
�
+dnn/head/metrics/auc_precision_recall/Sum_3Sum+dnn/head/metrics/auc_precision_recall/mul_3=dnn/head/metrics/auc_precision_recall/Sum_3/reduction_indices*
	keep_dims( *

Tidx0*
T0*
_output_shapes	
:�
�
1dnn/head/metrics/auc_precision_recall/AssignAdd_3	AssignAdd5dnn/head/metrics/auc_precision_recall/false_positives+dnn/head/metrics/auc_precision_recall/Sum_3*
_output_shapes	
:�*
use_locking( *
T0*H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_positives
p
+dnn/head/metrics/auc_precision_recall/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *�7�5
�
)dnn/head/metrics/auc_precision_recall/addAdd9dnn/head/metrics/auc_precision_recall/true_positives/read+dnn/head/metrics/auc_precision_recall/add/y*
_output_shapes	
:�*
T0
�
+dnn/head/metrics/auc_precision_recall/add_1Add9dnn/head/metrics/auc_precision_recall/true_positives/read:dnn/head/metrics/auc_precision_recall/false_negatives/read*
T0*
_output_shapes	
:�
r
-dnn/head/metrics/auc_precision_recall/add_2/yConst*
dtype0*
_output_shapes
: *
valueB
 *�7�5
�
+dnn/head/metrics/auc_precision_recall/add_2Add+dnn/head/metrics/auc_precision_recall/add_1-dnn/head/metrics/auc_precision_recall/add_2/y*
T0*
_output_shapes	
:�
�
)dnn/head/metrics/auc_precision_recall/divRealDiv)dnn/head/metrics/auc_precision_recall/add+dnn/head/metrics/auc_precision_recall/add_2*
_output_shapes	
:�*
T0
r
-dnn/head/metrics/auc_precision_recall/add_3/yConst*
valueB
 *�7�5*
dtype0*
_output_shapes
: 
�
+dnn/head/metrics/auc_precision_recall/add_3Add9dnn/head/metrics/auc_precision_recall/true_positives/read-dnn/head/metrics/auc_precision_recall/add_3/y*
_output_shapes	
:�*
T0
�
+dnn/head/metrics/auc_precision_recall/add_4Add9dnn/head/metrics/auc_precision_recall/true_positives/read:dnn/head/metrics/auc_precision_recall/false_positives/read*
_output_shapes	
:�*
T0
r
-dnn/head/metrics/auc_precision_recall/add_5/yConst*
dtype0*
_output_shapes
: *
valueB
 *�7�5
�
+dnn/head/metrics/auc_precision_recall/add_5Add+dnn/head/metrics/auc_precision_recall/add_4-dnn/head/metrics/auc_precision_recall/add_5/y*
T0*
_output_shapes	
:�
�
+dnn/head/metrics/auc_precision_recall/div_1RealDiv+dnn/head/metrics/auc_precision_recall/add_3+dnn/head/metrics/auc_precision_recall/add_5*
_output_shapes	
:�*
T0
�
;dnn/head/metrics/auc_precision_recall/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
=dnn/head/metrics/auc_precision_recall/strided_slice_1/stack_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
=dnn/head/metrics/auc_precision_recall/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
5dnn/head/metrics/auc_precision_recall/strided_slice_1StridedSlice)dnn/head/metrics/auc_precision_recall/div;dnn/head/metrics/auc_precision_recall/strided_slice_1/stack=dnn/head/metrics/auc_precision_recall/strided_slice_1/stack_1=dnn/head/metrics/auc_precision_recall/strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask *
_output_shapes	
:�
�
;dnn/head/metrics/auc_precision_recall/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
�
=dnn/head/metrics/auc_precision_recall/strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
=dnn/head/metrics/auc_precision_recall/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
5dnn/head/metrics/auc_precision_recall/strided_slice_2StridedSlice)dnn/head/metrics/auc_precision_recall/div;dnn/head/metrics/auc_precision_recall/strided_slice_2/stack=dnn/head/metrics/auc_precision_recall/strided_slice_2/stack_1=dnn/head/metrics/auc_precision_recall/strided_slice_2/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes	
:�*
T0*
Index0
�
)dnn/head/metrics/auc_precision_recall/subSub5dnn/head/metrics/auc_precision_recall/strided_slice_15dnn/head/metrics/auc_precision_recall/strided_slice_2*
_output_shapes	
:�*
T0
�
;dnn/head/metrics/auc_precision_recall/strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
=dnn/head/metrics/auc_precision_recall/strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB:�
�
=dnn/head/metrics/auc_precision_recall/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
5dnn/head/metrics/auc_precision_recall/strided_slice_3StridedSlice+dnn/head/metrics/auc_precision_recall/div_1;dnn/head/metrics/auc_precision_recall/strided_slice_3/stack=dnn/head/metrics/auc_precision_recall/strided_slice_3/stack_1=dnn/head/metrics/auc_precision_recall/strided_slice_3/stack_2*
end_mask *
_output_shapes	
:�*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask 
�
;dnn/head/metrics/auc_precision_recall/strided_slice_4/stackConst*
valueB:*
dtype0*
_output_shapes
:
�
=dnn/head/metrics/auc_precision_recall/strided_slice_4/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
�
=dnn/head/metrics/auc_precision_recall/strided_slice_4/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
5dnn/head/metrics/auc_precision_recall/strided_slice_4StridedSlice+dnn/head/metrics/auc_precision_recall/div_1;dnn/head/metrics/auc_precision_recall/strided_slice_4/stack=dnn/head/metrics/auc_precision_recall/strided_slice_4/stack_1=dnn/head/metrics/auc_precision_recall/strided_slice_4/stack_2*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask*
_output_shapes	
:�
�
+dnn/head/metrics/auc_precision_recall/add_6Add5dnn/head/metrics/auc_precision_recall/strided_slice_35dnn/head/metrics/auc_precision_recall/strided_slice_4*
T0*
_output_shapes	
:�
t
/dnn/head/metrics/auc_precision_recall/truediv/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
-dnn/head/metrics/auc_precision_recall/truedivRealDiv+dnn/head/metrics/auc_precision_recall/add_6/dnn/head/metrics/auc_precision_recall/truediv/y*
T0*
_output_shapes	
:�
�
)dnn/head/metrics/auc_precision_recall/MulMul)dnn/head/metrics/auc_precision_recall/sub-dnn/head/metrics/auc_precision_recall/truediv*
T0*
_output_shapes	
:�
w
-dnn/head/metrics/auc_precision_recall/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
�
+dnn/head/metrics/auc_precision_recall/valueSum)dnn/head/metrics/auc_precision_recall/Mul-dnn/head/metrics/auc_precision_recall/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
r
-dnn/head/metrics/auc_precision_recall/add_7/yConst*
valueB
 *�7�5*
dtype0*
_output_shapes
: 
�
+dnn/head/metrics/auc_precision_recall/add_7Add/dnn/head/metrics/auc_precision_recall/AssignAdd-dnn/head/metrics/auc_precision_recall/add_7/y*
T0*
_output_shapes	
:�
�
+dnn/head/metrics/auc_precision_recall/add_8Add/dnn/head/metrics/auc_precision_recall/AssignAdd1dnn/head/metrics/auc_precision_recall/AssignAdd_1*
_output_shapes	
:�*
T0
r
-dnn/head/metrics/auc_precision_recall/add_9/yConst*
valueB
 *�7�5*
dtype0*
_output_shapes
: 
�
+dnn/head/metrics/auc_precision_recall/add_9Add+dnn/head/metrics/auc_precision_recall/add_8-dnn/head/metrics/auc_precision_recall/add_9/y*
T0*
_output_shapes	
:�
�
+dnn/head/metrics/auc_precision_recall/div_2RealDiv+dnn/head/metrics/auc_precision_recall/add_7+dnn/head/metrics/auc_precision_recall/add_9*
T0*
_output_shapes	
:�
s
.dnn/head/metrics/auc_precision_recall/add_10/yConst*
valueB
 *�7�5*
dtype0*
_output_shapes
: 
�
,dnn/head/metrics/auc_precision_recall/add_10Add/dnn/head/metrics/auc_precision_recall/AssignAdd.dnn/head/metrics/auc_precision_recall/add_10/y*
T0*
_output_shapes	
:�
�
,dnn/head/metrics/auc_precision_recall/add_11Add/dnn/head/metrics/auc_precision_recall/AssignAdd1dnn/head/metrics/auc_precision_recall/AssignAdd_3*
_output_shapes	
:�*
T0
s
.dnn/head/metrics/auc_precision_recall/add_12/yConst*
_output_shapes
: *
valueB
 *�7�5*
dtype0
�
,dnn/head/metrics/auc_precision_recall/add_12Add,dnn/head/metrics/auc_precision_recall/add_11.dnn/head/metrics/auc_precision_recall/add_12/y*
_output_shapes	
:�*
T0
�
+dnn/head/metrics/auc_precision_recall/div_3RealDiv,dnn/head/metrics/auc_precision_recall/add_10,dnn/head/metrics/auc_precision_recall/add_12*
T0*
_output_shapes	
:�
�
;dnn/head/metrics/auc_precision_recall/strided_slice_5/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
=dnn/head/metrics/auc_precision_recall/strided_slice_5/stack_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
=dnn/head/metrics/auc_precision_recall/strided_slice_5/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
5dnn/head/metrics/auc_precision_recall/strided_slice_5StridedSlice+dnn/head/metrics/auc_precision_recall/div_2;dnn/head/metrics/auc_precision_recall/strided_slice_5/stack=dnn/head/metrics/auc_precision_recall/strided_slice_5/stack_1=dnn/head/metrics/auc_precision_recall/strided_slice_5/stack_2*
_output_shapes	
:�*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask 
�
;dnn/head/metrics/auc_precision_recall/strided_slice_6/stackConst*
valueB:*
dtype0*
_output_shapes
:
�
=dnn/head/metrics/auc_precision_recall/strided_slice_6/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
�
=dnn/head/metrics/auc_precision_recall/strided_slice_6/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
5dnn/head/metrics/auc_precision_recall/strided_slice_6StridedSlice+dnn/head/metrics/auc_precision_recall/div_2;dnn/head/metrics/auc_precision_recall/strided_slice_6/stack=dnn/head/metrics/auc_precision_recall/strided_slice_6/stack_1=dnn/head/metrics/auc_precision_recall/strided_slice_6/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes	
:�
�
+dnn/head/metrics/auc_precision_recall/sub_1Sub5dnn/head/metrics/auc_precision_recall/strided_slice_55dnn/head/metrics/auc_precision_recall/strided_slice_6*
T0*
_output_shapes	
:�
�
;dnn/head/metrics/auc_precision_recall/strided_slice_7/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
=dnn/head/metrics/auc_precision_recall/strided_slice_7/stack_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
=dnn/head/metrics/auc_precision_recall/strided_slice_7/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
5dnn/head/metrics/auc_precision_recall/strided_slice_7StridedSlice+dnn/head/metrics/auc_precision_recall/div_3;dnn/head/metrics/auc_precision_recall/strided_slice_7/stack=dnn/head/metrics/auc_precision_recall/strided_slice_7/stack_1=dnn/head/metrics/auc_precision_recall/strided_slice_7/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes	
:�*
Index0*
T0
�
;dnn/head/metrics/auc_precision_recall/strided_slice_8/stackConst*
_output_shapes
:*
valueB:*
dtype0
�
=dnn/head/metrics/auc_precision_recall/strided_slice_8/stack_1Const*
_output_shapes
:*
valueB: *
dtype0
�
=dnn/head/metrics/auc_precision_recall/strided_slice_8/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
5dnn/head/metrics/auc_precision_recall/strided_slice_8StridedSlice+dnn/head/metrics/auc_precision_recall/div_3;dnn/head/metrics/auc_precision_recall/strided_slice_8/stack=dnn/head/metrics/auc_precision_recall/strided_slice_8/stack_1=dnn/head/metrics/auc_precision_recall/strided_slice_8/stack_2*
end_mask*
_output_shapes	
:�*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask 
�
,dnn/head/metrics/auc_precision_recall/add_13Add5dnn/head/metrics/auc_precision_recall/strided_slice_75dnn/head/metrics/auc_precision_recall/strided_slice_8*
T0*
_output_shapes	
:�
v
1dnn/head/metrics/auc_precision_recall/truediv_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *   @
�
/dnn/head/metrics/auc_precision_recall/truediv_1RealDiv,dnn/head/metrics/auc_precision_recall/add_131dnn/head/metrics/auc_precision_recall/truediv_1/y*
T0*
_output_shapes	
:�
�
+dnn/head/metrics/auc_precision_recall/Mul_1Mul+dnn/head/metrics/auc_precision_recall/sub_1/dnn/head/metrics/auc_precision_recall/truediv_1*
T0*
_output_shapes	
:�
w
-dnn/head/metrics/auc_precision_recall/Const_2Const*
_output_shapes
:*
valueB: *
dtype0
�
/dnn/head/metrics/auc_precision_recall/update_opSum+dnn/head/metrics/auc_precision_recall/Mul_1-dnn/head/metrics/auc_precision_recall/Const_2*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
mean/total/Initializer/zerosConst*
_class
loc:@mean/total*
valueB
 *    *
dtype0*
_output_shapes
: 
�

mean/total
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@mean/total
�
mean/total/AssignAssign
mean/totalmean/total/Initializer/zeros*
use_locking(*
T0*
_class
loc:@mean/total*
validate_shape(*
_output_shapes
: 
g
mean/total/readIdentity
mean/total*
T0*
_class
loc:@mean/total*
_output_shapes
: 
�
mean/count/Initializer/zerosConst*
_class
loc:@mean/count*
valueB
 *    *
dtype0*
_output_shapes
: 
�

mean/count
VariableV2*
shared_name *
_class
loc:@mean/count*
	container *
shape: *
dtype0*
_output_shapes
: 
�
mean/count/AssignAssign
mean/countmean/count/Initializer/zeros*
_class
loc:@mean/count*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
g
mean/count/readIdentity
mean/count*
_output_shapes
: *
T0*
_class
loc:@mean/count
K
	mean/SizeConst*
_output_shapes
: *
value	B :*
dtype0
Q
mean/ToFloat_1Cast	mean/Size*

SrcT0*
_output_shapes
: *

DstT0
M

mean/ConstConst*
valueB *
dtype0*
_output_shapes
: 
u
mean/SumSumdnn/head/weighted_loss/Sum
mean/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
mean/AssignAdd	AssignAdd
mean/totalmean/Sum*
_output_shapes
: *
use_locking( *
T0*
_class
loc:@mean/total
�
mean/AssignAdd_1	AssignAdd
mean/countmean/ToFloat_1^dnn/head/weighted_loss/Sum*
use_locking( *
T0*
_class
loc:@mean/count*
_output_shapes
: 
Z
mean/truedivRealDivmean/total/readmean/count/read*
T0*
_output_shapes
: 
T
mean/zeros_likeConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
mean/GreaterGreatermean/count/readmean/zeros_like*
T0*
_output_shapes
: 
b

mean/valueSelectmean/Greatermean/truedivmean/zeros_like*
_output_shapes
: *
T0
\
mean/truediv_1RealDivmean/AssignAddmean/AssignAdd_1*
T0*
_output_shapes
: 
V
mean/zeros_like_1Const*
valueB
 *    *
dtype0*
_output_shapes
: 
_
mean/Greater_1Greatermean/AssignAdd_1mean/zeros_like_1*
T0*
_output_shapes
: 
l
mean/update_opSelectmean/Greater_1mean/truediv_1mean/zeros_like_1*
_output_shapes
: *
T0
�

group_depsNoOp$^dnn/head/metrics/accuracy/update_op-^dnn/head/metrics/accuracy_baseline/update_op^dnn/head/metrics/auc/update_op0^dnn/head/metrics/auc_precision_recall/update_op(^dnn/head/metrics/average_loss/update_op&^dnn/head/metrics/label/mean/update_op%^dnn/head/metrics/precision/update_op+^dnn/head/metrics/prediction/mean/update_op"^dnn/head/metrics/recall/update_op^mean/update_op
{
eval_step/Initializer/zerosConst*
_class
loc:@eval_step*
value	B	 R *
dtype0	*
_output_shapes
: 
�
	eval_step
VariableV2*
dtype0	*
_output_shapes
: *
shared_name *
_class
loc:@eval_step*
	container *
shape: 
�
eval_step/AssignAssign	eval_stepeval_step/Initializer/zeros*
use_locking(*
T0	*
_class
loc:@eval_step*
validate_shape(*
_output_shapes
: 
d
eval_step/readIdentity	eval_step*
T0	*
_class
loc:@eval_step*
_output_shapes
: 
Q
AssignAdd/valueConst*
value	B	 R*
dtype0	*
_output_shapes
: 
�
	AssignAdd	AssignAdd	eval_stepAssignAdd/value*
_output_shapes
: *
use_locking(*
T0	*
_class
loc:@eval_step
U
readIdentity	eval_step
^AssignAdd^group_deps*
T0	*
_output_shapes
: 
;
IdentityIdentityread*
T0	*
_output_shapes
: 
�
initNoOp%^dnn/hiddenlayer_0/bias/part_0/Assign'^dnn/hiddenlayer_0/kernel/part_0/Assign%^dnn/hiddenlayer_1/bias/part_0/Assign'^dnn/hiddenlayer_1/kernel/part_0/Assign^dnn/logits/bias/part_0/Assign ^dnn/logits/kernel/part_0/Assign^global_step/Assign

init_1NoOp
$
group_deps_1NoOp^init^init_1
�
4report_uninitialized_variables/IsVariableInitializedIsVariableInitializedglobal_step*
_output_shapes
: *
_class
loc:@global_step*
dtype0	
�
6report_uninitialized_variables/IsVariableInitialized_1IsVariableInitializeddnn/hiddenlayer_0/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_2IsVariableInitializeddnn/hiddenlayer_0/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_3IsVariableInitializeddnn/hiddenlayer_1/kernel/part_0*
_output_shapes
: *2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0
�
6report_uninitialized_variables/IsVariableInitialized_4IsVariableInitializeddnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes
: *0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0
�
6report_uninitialized_variables/IsVariableInitialized_5IsVariableInitializeddnn/logits/kernel/part_0*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_6IsVariableInitializeddnn/logits/bias/part_0*)
_class
loc:@dnn/logits/bias/part_0*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_7IsVariableInitialized!dnn/head/metrics/label/mean/total*4
_class*
(&loc:@dnn/head/metrics/label/mean/total*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_8IsVariableInitialized!dnn/head/metrics/label/mean/count*4
_class*
(&loc:@dnn/head/metrics/label/mean/count*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_9IsVariableInitialized#dnn/head/metrics/average_loss/total*6
_class,
*(loc:@dnn/head/metrics/average_loss/total*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_10IsVariableInitialized#dnn/head/metrics/average_loss/count*6
_class,
*(loc:@dnn/head/metrics/average_loss/count*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_11IsVariableInitializeddnn/head/metrics/accuracy/total*2
_class(
&$loc:@dnn/head/metrics/accuracy/total*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_12IsVariableInitializeddnn/head/metrics/accuracy/count*2
_class(
&$loc:@dnn/head/metrics/accuracy/count*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_13IsVariableInitialized/dnn/head/metrics/precision/true_positives/count*
dtype0*
_output_shapes
: *B
_class8
64loc:@dnn/head/metrics/precision/true_positives/count
�
7report_uninitialized_variables/IsVariableInitialized_14IsVariableInitialized0dnn/head/metrics/precision/false_positives/count*C
_class9
75loc:@dnn/head/metrics/precision/false_positives/count*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_15IsVariableInitialized,dnn/head/metrics/recall/true_positives/count*?
_class5
31loc:@dnn/head/metrics/recall/true_positives/count*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_16IsVariableInitialized-dnn/head/metrics/recall/false_negatives/count*@
_class6
42loc:@dnn/head/metrics/recall/false_negatives/count*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_17IsVariableInitialized&dnn/head/metrics/prediction/mean/total*
_output_shapes
: *9
_class/
-+loc:@dnn/head/metrics/prediction/mean/total*
dtype0
�
7report_uninitialized_variables/IsVariableInitialized_18IsVariableInitialized&dnn/head/metrics/prediction/mean/count*9
_class/
-+loc:@dnn/head/metrics/prediction/mean/count*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_19IsVariableInitialized#dnn/head/metrics/auc/true_positives*6
_class,
*(loc:@dnn/head/metrics/auc/true_positives*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_20IsVariableInitialized$dnn/head/metrics/auc/false_negatives*
_output_shapes
: *7
_class-
+)loc:@dnn/head/metrics/auc/false_negatives*
dtype0
�
7report_uninitialized_variables/IsVariableInitialized_21IsVariableInitialized#dnn/head/metrics/auc/true_negatives*6
_class,
*(loc:@dnn/head/metrics/auc/true_negatives*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_22IsVariableInitialized$dnn/head/metrics/auc/false_positives*7
_class-
+)loc:@dnn/head/metrics/auc/false_positives*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_23IsVariableInitialized4dnn/head/metrics/auc_precision_recall/true_positives*G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_positives*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_24IsVariableInitialized5dnn/head/metrics/auc_precision_recall/false_negatives*H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_negatives*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_25IsVariableInitialized4dnn/head/metrics/auc_precision_recall/true_negatives*G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_negatives*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_26IsVariableInitialized5dnn/head/metrics/auc_precision_recall/false_positives*H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_positives*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_27IsVariableInitialized
mean/total*
_output_shapes
: *
_class
loc:@mean/total*
dtype0
�
7report_uninitialized_variables/IsVariableInitialized_28IsVariableInitialized
mean/count*
_class
loc:@mean/count*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_29IsVariableInitialized	eval_step*
_class
loc:@eval_step*
dtype0	*
_output_shapes
: 
�
$report_uninitialized_variables/stackPack4report_uninitialized_variables/IsVariableInitialized6report_uninitialized_variables/IsVariableInitialized_16report_uninitialized_variables/IsVariableInitialized_26report_uninitialized_variables/IsVariableInitialized_36report_uninitialized_variables/IsVariableInitialized_46report_uninitialized_variables/IsVariableInitialized_56report_uninitialized_variables/IsVariableInitialized_66report_uninitialized_variables/IsVariableInitialized_76report_uninitialized_variables/IsVariableInitialized_86report_uninitialized_variables/IsVariableInitialized_97report_uninitialized_variables/IsVariableInitialized_107report_uninitialized_variables/IsVariableInitialized_117report_uninitialized_variables/IsVariableInitialized_127report_uninitialized_variables/IsVariableInitialized_137report_uninitialized_variables/IsVariableInitialized_147report_uninitialized_variables/IsVariableInitialized_157report_uninitialized_variables/IsVariableInitialized_167report_uninitialized_variables/IsVariableInitialized_177report_uninitialized_variables/IsVariableInitialized_187report_uninitialized_variables/IsVariableInitialized_197report_uninitialized_variables/IsVariableInitialized_207report_uninitialized_variables/IsVariableInitialized_217report_uninitialized_variables/IsVariableInitialized_227report_uninitialized_variables/IsVariableInitialized_237report_uninitialized_variables/IsVariableInitialized_247report_uninitialized_variables/IsVariableInitialized_257report_uninitialized_variables/IsVariableInitialized_267report_uninitialized_variables/IsVariableInitialized_277report_uninitialized_variables/IsVariableInitialized_287report_uninitialized_variables/IsVariableInitialized_29"/device:CPU:0*
T0
*

axis *
N*
_output_shapes
:
�
)report_uninitialized_variables/LogicalNot
LogicalNot$report_uninitialized_variables/stack"/device:CPU:0*
_output_shapes
:
�	
$report_uninitialized_variables/ConstConst"/device:CPU:0*
_output_shapes
:*�
value�B�Bglobal_stepBdnn/hiddenlayer_0/kernel/part_0Bdnn/hiddenlayer_0/bias/part_0Bdnn/hiddenlayer_1/kernel/part_0Bdnn/hiddenlayer_1/bias/part_0Bdnn/logits/kernel/part_0Bdnn/logits/bias/part_0B!dnn/head/metrics/label/mean/totalB!dnn/head/metrics/label/mean/countB#dnn/head/metrics/average_loss/totalB#dnn/head/metrics/average_loss/countBdnn/head/metrics/accuracy/totalBdnn/head/metrics/accuracy/countB/dnn/head/metrics/precision/true_positives/countB0dnn/head/metrics/precision/false_positives/countB,dnn/head/metrics/recall/true_positives/countB-dnn/head/metrics/recall/false_negatives/countB&dnn/head/metrics/prediction/mean/totalB&dnn/head/metrics/prediction/mean/countB#dnn/head/metrics/auc/true_positivesB$dnn/head/metrics/auc/false_negativesB#dnn/head/metrics/auc/true_negativesB$dnn/head/metrics/auc/false_positivesB4dnn/head/metrics/auc_precision_recall/true_positivesB5dnn/head/metrics/auc_precision_recall/false_negativesB4dnn/head/metrics/auc_precision_recall/true_negativesB5dnn/head/metrics/auc_precision_recall/false_positivesB
mean/totalB
mean/countB	eval_step*
dtype0
�
1report_uninitialized_variables/boolean_mask/ShapeConst"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
?report_uninitialized_variables/boolean_mask/strided_slice/stackConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_1Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
9report_uninitialized_variables/boolean_mask/strided_sliceStridedSlice1report_uninitialized_variables/boolean_mask/Shape?report_uninitialized_variables/boolean_mask/strided_slice/stackAreport_uninitialized_variables/boolean_mask/strided_slice/stack_1Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2"/device:CPU:0*
_output_shapes
:*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
�
Breport_uninitialized_variables/boolean_mask/Prod/reduction_indicesConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
0report_uninitialized_variables/boolean_mask/ProdProd9report_uninitialized_variables/boolean_mask/strided_sliceBreport_uninitialized_variables/boolean_mask/Prod/reduction_indices"/device:CPU:0*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
3report_uninitialized_variables/boolean_mask/Shape_1Const"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:
�
Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Const"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
;report_uninitialized_variables/boolean_mask/strided_slice_1StridedSlice3report_uninitialized_variables/boolean_mask/Shape_1Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackCreport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2"/device:CPU:0*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask *
_output_shapes
: 
�
3report_uninitialized_variables/boolean_mask/Shape_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
Areport_uninitialized_variables/boolean_mask/strided_slice_2/stackConst"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
Creport_uninitialized_variables/boolean_mask/strided_slice_2/stack_1Const"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
Creport_uninitialized_variables/boolean_mask/strided_slice_2/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
;report_uninitialized_variables/boolean_mask/strided_slice_2StridedSlice3report_uninitialized_variables/boolean_mask/Shape_2Areport_uninitialized_variables/boolean_mask/strided_slice_2/stackCreport_uninitialized_variables/boolean_mask/strided_slice_2/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_2/stack_2"/device:CPU:0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
: *
T0*
Index0
�
;report_uninitialized_variables/boolean_mask/concat/values_1Pack0report_uninitialized_variables/boolean_mask/Prod"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:
�
7report_uninitialized_variables/boolean_mask/concat/axisConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
2report_uninitialized_variables/boolean_mask/concatConcatV2;report_uninitialized_variables/boolean_mask/strided_slice_1;report_uninitialized_variables/boolean_mask/concat/values_1;report_uninitialized_variables/boolean_mask/strided_slice_27report_uninitialized_variables/boolean_mask/concat/axis"/device:CPU:0*
N*
_output_shapes
:*

Tidx0*
T0
�
3report_uninitialized_variables/boolean_mask/ReshapeReshape$report_uninitialized_variables/Const2report_uninitialized_variables/boolean_mask/concat"/device:CPU:0*
_output_shapes
:*
T0*
Tshape0
�
;report_uninitialized_variables/boolean_mask/Reshape_1/shapeConst"/device:CPU:0*
valueB:
���������*
dtype0*
_output_shapes
:
�
5report_uninitialized_variables/boolean_mask/Reshape_1Reshape)report_uninitialized_variables/LogicalNot;report_uninitialized_variables/boolean_mask/Reshape_1/shape"/device:CPU:0*
T0
*
Tshape0*
_output_shapes
:
�
1report_uninitialized_variables/boolean_mask/WhereWhere5report_uninitialized_variables/boolean_mask/Reshape_1"/device:CPU:0*
T0
*'
_output_shapes
:���������
�
3report_uninitialized_variables/boolean_mask/SqueezeSqueeze1report_uninitialized_variables/boolean_mask/Where"/device:CPU:0*#
_output_shapes
:���������*
squeeze_dims
*
T0	
�
9report_uninitialized_variables/boolean_mask/GatherV2/axisConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
4report_uninitialized_variables/boolean_mask/GatherV2GatherV23report_uninitialized_variables/boolean_mask/Reshape3report_uninitialized_variables/boolean_mask/Squeeze9report_uninitialized_variables/boolean_mask/GatherV2/axis"/device:CPU:0*
Tindices0	*
Tparams0*#
_output_shapes
:���������*
Taxis0
v
$report_uninitialized_resources/ConstConst"/device:CPU:0*
_output_shapes
: *
valueB *
dtype0
M
concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
�
concatConcatV24report_uninitialized_variables/boolean_mask/GatherV2$report_uninitialized_resources/Constconcat/axis*
N*#
_output_shapes
:���������*

Tidx0*
T0
�
6report_uninitialized_variables_1/IsVariableInitializedIsVariableInitializedglobal_step*
dtype0	*
_output_shapes
: *
_class
loc:@global_step
�
8report_uninitialized_variables_1/IsVariableInitialized_1IsVariableInitializeddnn/hiddenlayer_0/kernel/part_0*
_output_shapes
: *2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0
�
8report_uninitialized_variables_1/IsVariableInitialized_2IsVariableInitializeddnn/hiddenlayer_0/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_3IsVariableInitializeddnn/hiddenlayer_1/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_4IsVariableInitializeddnn/hiddenlayer_1/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_5IsVariableInitializeddnn/logits/kernel/part_0*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_6IsVariableInitializeddnn/logits/bias/part_0*)
_class
loc:@dnn/logits/bias/part_0*
dtype0*
_output_shapes
: 
�
&report_uninitialized_variables_1/stackPack6report_uninitialized_variables_1/IsVariableInitialized8report_uninitialized_variables_1/IsVariableInitialized_18report_uninitialized_variables_1/IsVariableInitialized_28report_uninitialized_variables_1/IsVariableInitialized_38report_uninitialized_variables_1/IsVariableInitialized_48report_uninitialized_variables_1/IsVariableInitialized_58report_uninitialized_variables_1/IsVariableInitialized_6"/device:CPU:0*
T0
*

axis *
N*
_output_shapes
:
�
+report_uninitialized_variables_1/LogicalNot
LogicalNot&report_uninitialized_variables_1/stack"/device:CPU:0*
_output_shapes
:
�
&report_uninitialized_variables_1/ConstConst"/device:CPU:0*�
value�B�Bglobal_stepBdnn/hiddenlayer_0/kernel/part_0Bdnn/hiddenlayer_0/bias/part_0Bdnn/hiddenlayer_1/kernel/part_0Bdnn/hiddenlayer_1/bias/part_0Bdnn/logits/kernel/part_0Bdnn/logits/bias/part_0*
dtype0*
_output_shapes
:
�
3report_uninitialized_variables_1/boolean_mask/ShapeConst"/device:CPU:0*
_output_shapes
:*
valueB:*
dtype0
�
Areport_uninitialized_variables_1/boolean_mask/strided_slice/stackConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
;report_uninitialized_variables_1/boolean_mask/strided_sliceStridedSlice3report_uninitialized_variables_1/boolean_mask/ShapeAreport_uninitialized_variables_1/boolean_mask/strided_slice/stackCreport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2"/device:CPU:0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
:*
T0*
Index0
�
Dreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indicesConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
2report_uninitialized_variables_1/boolean_mask/ProdProd;report_uninitialized_variables_1/boolean_mask/strided_sliceDreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indices"/device:CPU:0*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
5report_uninitialized_variables_1/boolean_mask/Shape_1Const"/device:CPU:0*
_output_shapes
:*
valueB:*
dtype0
�
Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Const"/device:CPU:0*
_output_shapes
:*
valueB: *
dtype0
�
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
=report_uninitialized_variables_1/boolean_mask/strided_slice_1StridedSlice5report_uninitialized_variables_1/boolean_mask/Shape_1Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackEreport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2"/device:CPU:0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
�
5report_uninitialized_variables_1/boolean_mask/Shape_2Const"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:
�
Creport_uninitialized_variables_1/boolean_mask/strided_slice_2/stackConst"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_1Const"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_2Const"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:
�
=report_uninitialized_variables_1/boolean_mask/strided_slice_2StridedSlice5report_uninitialized_variables_1/boolean_mask/Shape_2Creport_uninitialized_variables_1/boolean_mask/strided_slice_2/stackEreport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_1Ereport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_2"/device:CPU:0*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
: *
Index0*
T0*
shrink_axis_mask 
�
=report_uninitialized_variables_1/boolean_mask/concat/values_1Pack2report_uninitialized_variables_1/boolean_mask/Prod"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:
�
9report_uninitialized_variables_1/boolean_mask/concat/axisConst"/device:CPU:0*
_output_shapes
: *
value	B : *
dtype0
�
4report_uninitialized_variables_1/boolean_mask/concatConcatV2=report_uninitialized_variables_1/boolean_mask/strided_slice_1=report_uninitialized_variables_1/boolean_mask/concat/values_1=report_uninitialized_variables_1/boolean_mask/strided_slice_29report_uninitialized_variables_1/boolean_mask/concat/axis"/device:CPU:0*

Tidx0*
T0*
N*
_output_shapes
:
�
5report_uninitialized_variables_1/boolean_mask/ReshapeReshape&report_uninitialized_variables_1/Const4report_uninitialized_variables_1/boolean_mask/concat"/device:CPU:0*
Tshape0*
_output_shapes
:*
T0
�
=report_uninitialized_variables_1/boolean_mask/Reshape_1/shapeConst"/device:CPU:0*
valueB:
���������*
dtype0*
_output_shapes
:
�
7report_uninitialized_variables_1/boolean_mask/Reshape_1Reshape+report_uninitialized_variables_1/LogicalNot=report_uninitialized_variables_1/boolean_mask/Reshape_1/shape"/device:CPU:0*
T0
*
Tshape0*
_output_shapes
:
�
3report_uninitialized_variables_1/boolean_mask/WhereWhere7report_uninitialized_variables_1/boolean_mask/Reshape_1"/device:CPU:0*'
_output_shapes
:���������*
T0

�
5report_uninitialized_variables_1/boolean_mask/SqueezeSqueeze3report_uninitialized_variables_1/boolean_mask/Where"/device:CPU:0*#
_output_shapes
:���������*
squeeze_dims
*
T0	
�
;report_uninitialized_variables_1/boolean_mask/GatherV2/axisConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables_1/boolean_mask/GatherV2GatherV25report_uninitialized_variables_1/boolean_mask/Reshape5report_uninitialized_variables_1/boolean_mask/Squeeze;report_uninitialized_variables_1/boolean_mask/GatherV2/axis"/device:CPU:0*#
_output_shapes
:���������*
Taxis0*
Tindices0	*
Tparams0
�
init_2NoOp'^dnn/head/metrics/accuracy/count/Assign'^dnn/head/metrics/accuracy/total/Assign,^dnn/head/metrics/auc/false_negatives/Assign,^dnn/head/metrics/auc/false_positives/Assign+^dnn/head/metrics/auc/true_negatives/Assign+^dnn/head/metrics/auc/true_positives/Assign=^dnn/head/metrics/auc_precision_recall/false_negatives/Assign=^dnn/head/metrics/auc_precision_recall/false_positives/Assign<^dnn/head/metrics/auc_precision_recall/true_negatives/Assign<^dnn/head/metrics/auc_precision_recall/true_positives/Assign+^dnn/head/metrics/average_loss/count/Assign+^dnn/head/metrics/average_loss/total/Assign)^dnn/head/metrics/label/mean/count/Assign)^dnn/head/metrics/label/mean/total/Assign8^dnn/head/metrics/precision/false_positives/count/Assign7^dnn/head/metrics/precision/true_positives/count/Assign.^dnn/head/metrics/prediction/mean/count/Assign.^dnn/head/metrics/prediction/mean/total/Assign5^dnn/head/metrics/recall/false_negatives/count/Assign4^dnn/head/metrics/recall/true_positives/count/Assign^eval_step/Assign^mean/count/Assign^mean/total/Assign

init_all_tablesNoOp

init_3NoOp
8
group_deps_2NoOp^init_2^init_3^init_all_tables
�
Merge/MergeSummaryMergeSummaryHenqueue_input/queue/enqueue_input/fifo_queuefraction_over_0_of_1000_full-dnn/dnn/hiddenlayer_0/fraction_of_zero_values dnn/dnn/hiddenlayer_0/activation-dnn/dnn/hiddenlayer_1/fraction_of_zero_values dnn/dnn/hiddenlayer_1/activation&dnn/dnn/logits/fraction_of_zero_valuesdnn/dnn/logits/activation*
N*
_output_shapes
: 
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
�
save/StringJoin/inputs_1Const*<
value3B1 B+_temp_2643c5b94d414b26b776ae06de9d9710/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : 
�
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst"/device:CPU:0*�
value�B�Bdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/logits/biasBdnn/logits/kernelBglobal_step*
dtype0*
_output_shapes
:
�
save/SaveV2/shape_and_slicesConst"/device:CPU:0*i
value`B^B	256 0,256B4096 256 0,4096:0,256B32 0,32B256 32 0,256:0,32B1 0,1B32 1 0,32:0,1B *
dtype0*
_output_shapes
:
�
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices"dnn/hiddenlayer_0/bias/part_0/read$dnn/hiddenlayer_0/kernel/part_0/read"dnn/hiddenlayer_1/bias/part_0/read$dnn/hiddenlayer_1/kernel/part_0/readdnn/logits/bias/part_0/readdnn/logits/kernel/part_0/readglobal_step"/device:CPU:0*
dtypes
	2	
�
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
�
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:
�
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(
�
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
�
save/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*�
value�B�Bdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/logits/biasBdnn/logits/kernelBglobal_step
�
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*i
value`B^B	256 0,256B4096 256 0,4096:0,256B32 0,32B256 32 0,256:0,32B1 0,1B32 1 0,32:0,1B 
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
	2	*L
_output_shapes:
8:�:
� �: :	� :: :
�
save/AssignAssigndnn/hiddenlayer_0/bias/part_0save/RestoreV2*
use_locking(*
T0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
validate_shape(*
_output_shapes	
:�
�
save/Assign_1Assigndnn/hiddenlayer_0/kernel/part_0save/RestoreV2:1*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
validate_shape(* 
_output_shapes
:
� �*
use_locking(
�
save/Assign_2Assigndnn/hiddenlayer_1/bias/part_0save/RestoreV2:2*
use_locking(*
T0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
validate_shape(*
_output_shapes
: 
�
save/Assign_3Assigndnn/hiddenlayer_1/kernel/part_0save/RestoreV2:3*
_output_shapes
:	� *
use_locking(*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
validate_shape(
�
save/Assign_4Assigndnn/logits/bias/part_0save/RestoreV2:4*
use_locking(*
T0*)
_class
loc:@dnn/logits/bias/part_0*
validate_shape(*
_output_shapes
:
�
save/Assign_5Assigndnn/logits/kernel/part_0save/RestoreV2:5*
_output_shapes

: *
use_locking(*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
validate_shape(
�
save/Assign_6Assignglobal_stepsave/RestoreV2:6*
_output_shapes
: *
use_locking(*
T0	*
_class
loc:@global_step*
validate_shape(
�
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6
-
save/restore_allNoOp^save/restore_shard""�	
trainable_variables�	�	
�
!dnn/hiddenlayer_0/kernel/part_0:0&dnn/hiddenlayer_0/kernel/part_0/Assign&dnn/hiddenlayer_0/kernel/part_0/read:0"*
dnn/hiddenlayer_0/kernel� �  "� �2<dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform:0
�
dnn/hiddenlayer_0/bias/part_0:0$dnn/hiddenlayer_0/bias/part_0/Assign$dnn/hiddenlayer_0/bias/part_0/read:0"#
dnn/hiddenlayer_0/bias� "�21dnn/hiddenlayer_0/bias/part_0/Initializer/zeros:0
�
!dnn/hiddenlayer_1/kernel/part_0:0&dnn/hiddenlayer_1/kernel/part_0/Assign&dnn/hiddenlayer_1/kernel/part_0/read:0"(
dnn/hiddenlayer_1/kernel�   "� 2<dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform:0
�
dnn/hiddenlayer_1/bias/part_0:0$dnn/hiddenlayer_1/bias/part_0/Assign$dnn/hiddenlayer_1/bias/part_0/read:0"!
dnn/hiddenlayer_1/bias  " 21dnn/hiddenlayer_1/bias/part_0/Initializer/zeros:0
�
dnn/logits/kernel/part_0:0dnn/logits/kernel/part_0/Assigndnn/logits/kernel/part_0/read:0"
dnn/logits/kernel   " 25dnn/logits/kernel/part_0/Initializer/random_uniform:0
�
dnn/logits/bias/part_0:0dnn/logits/bias/part_0/Assigndnn/logits/bias/part_0/read:0"
dnn/logits/bias "2*dnn/logits/bias/part_0/Initializer/zeros:0"�
	summaries�
�
Jenqueue_input/queue/enqueue_input/fifo_queuefraction_over_0_of_1000_full:0
/dnn/dnn/hiddenlayer_0/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_0/activation:0
/dnn/dnn/hiddenlayer_1/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_1/activation:0
(dnn/dnn/logits/fraction_of_zero_values:0
dnn/dnn/logits/activation:0"W
ready_for_local_init_op<
:
8report_uninitialized_variables_1/boolean_mask/GatherV2:0"
init_op

group_deps_1"
	eval_step

eval_step:0"��
cond_context����
�
>dnn/head/assert_range/assert_less/Assert/AssertGuard/cond_text>dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_id:0?dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_t:0 *�
Idnn/head/assert_range/assert_less/Assert/AssertGuard/control_dependency:0
>dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_id:0
?dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_t:0�
?dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_t:0?dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_t:0�
>dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_id:0>dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_id:0
�
@dnn/head/assert_range/assert_less/Assert/AssertGuard/cond_text_1>dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_id:0?dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f:0*�

dnn/head/ToFloat:0
dnn/head/assert_range/Const:0
'dnn/head/assert_range/assert_less/All:0
Ddnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/Switch:0
Fdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/Switch_1:0
Fdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/Switch_2:0
Ddnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_0:0
Ddnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_1:0
Ddnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_2:0
Ddnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_4:0
Kdnn/head/assert_range/assert_less/Assert/AssertGuard/control_dependency_1:0
>dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_id:0
?dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f:0\
dnn/head/ToFloat:0Fdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/Switch_1:0g
dnn/head/assert_range/Const:0Fdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/Switch_2:0o
'dnn/head/assert_range/assert_less/All:0Ddnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/Switch:0�
?dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f:0?dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f:0�
>dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_id:0>dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_id:0
�
Xdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/cond_textXdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_id:0Ydnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_t:0 *�
cdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/control_dependency:0
Xdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_id:0
Ydnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_t:0�
Ydnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_t:0Ydnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_t:0�
Xdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_id:0Xdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_id:0
�
Zdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/cond_text_1Xdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_id:0Ydnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_f:0*�
dnn/head/ToFloat:0
Adnn/head/assert_range/assert_non_negative/assert_less_equal/All:0
^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/Switch:0
`dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/Switch_1:0
^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0:0
^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1:0
^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2:0
ednn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/control_dependency_1:0
Xdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_id:0
Ydnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_f:0�
Adnn/head/assert_range/assert_non_negative/assert_less_equal/All:0^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/Switch:0�
Ydnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_f:0Ydnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_f:0�
Xdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_id:0Xdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_id:0v
dnn/head/ToFloat:0`dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/Switch_1:0
�

]dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/cond_text]dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_t:0 *�
Pdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalar:0
^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1:0
^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1:1
]dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0
^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_t:0�
]dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0]dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0�
^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_t:0^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_t:0�
Pdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalar:0^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1:1
�e
_dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/cond_text_1]dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_f:0*�-
udnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge:0
udnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge:1
vdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:0
vdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1
xdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
xdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:1
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:0
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:2
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:0
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:0
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:0
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:0
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:0
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:0
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:0
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:0
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:0
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:0
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:0
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:0
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch:0
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1:0
|dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0
wdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
xdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0
xdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0
]dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0
^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_f:0
Rdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/rank:0
Sdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/shape:0
Sdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/rank:0
Tdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/shape:0�
Sdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/shape:0�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0�
Sdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/rank:0�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1:0�
]dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0]dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0�
Tdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/shape:0�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0�
Rdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/rank:0�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch:0�
^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_f:0^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_f:02�(
�(
wdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_textwdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0xdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0 *�%
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:0
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:2
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:0
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:0
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:0
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:0
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:0
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:0
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:0
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:0
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:0
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:0
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:0
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:0
wdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
xdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0
Sdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/shape:0
Tdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/shape:0�
Tdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/shape:0�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1�
xdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0xdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0�
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0�
Sdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/shape:0�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1�
wdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0wdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0�
�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0�dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:02�
�
ydnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_text_1wdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0xdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0*�

xdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
xdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:1
|dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0
wdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
xdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0�
wdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0wdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0�
|dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0xdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0�
xdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0xdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0
�
Zdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/cond_textZdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0[dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t:0 *�
ednn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency:0
Zdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0
[dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t:0�
[dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t:0[dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t:0�
Zdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0Zdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0
�
\dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/cond_text_1Zdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0[dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f:0*�
`dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch:0
bdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1:0
bdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2:0
bdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3:0
`dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0:0
`dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1:0
`dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2:0
`dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4:0
`dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5:0
`dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7:0
gdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency_1:0
Zdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0
[dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f:0
Pdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalar:0
[dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge:0
Sdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/shape:0
Tdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/shape:0�
Pdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalar:0bdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3:0�
[dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge:0`dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch:0�
[dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f:0[dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f:0�
Zdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0Zdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0�
Sdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/shape:0bdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2:0�
Tdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/shape:0bdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1:0
�

bdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/cond_textbdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_t:0 *�
Udnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalar:0
cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1:0
cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1:1
bdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0
cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_t:0�
Udnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalar:0cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1:1�
bdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0bdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0�
cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_t:0cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_t:0
�i
ddnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/cond_text_1bdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_f:0*�/
zdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge:0
zdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge:1
{dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:0
{dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1
}dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
}dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:1
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:0
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:2
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:0
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:0
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:0
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:0
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:0
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:0
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:0
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:0
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:0
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:0
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:0
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:0
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch:0
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1:0
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0
|dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
}dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0
}dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0
bdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0
cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_f:0
Wdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/rank:0
Xdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/shape:0
Xdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/rank:0
Ydnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/shape:0�
Ydnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/shape:0�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0�
bdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0bdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0�
Xdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/shape:0�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0�
Xdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/rank:0�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1:0�
Wdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/rank:0�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch:0�
cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_f:0cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_f:02�)
�)
|dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_text|dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0}dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0 *�&
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:0
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:2
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:0
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:0
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:0
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:0
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:0
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:0
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:0
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:0
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:0
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:0
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:0
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:0
|dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
}dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0
Xdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/shape:0
Ydnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/shape:0�
Ydnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/shape:0�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1�
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0�
Xdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/shape:0�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1�
}dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0}dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0�
|dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0|dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0�
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:02�
�
~dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_text_1|dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0}dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0*�
}dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
}dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:1
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0
|dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
}dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0�
�dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0}dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0�
}dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0}dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0�
|dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0|dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
�
_dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/cond_text_dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0`dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t:0 *�
jdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency:0
_dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0
`dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t:0�
`dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t:0`dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t:0�
_dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0_dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0
�
adnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/cond_text_1_dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0`dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f:0*�
ednn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch:0
gdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1:0
gdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2:0
gdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3:0
ednn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0:0
ednn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1:0
ednn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2:0
ednn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4:0
ednn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5:0
ednn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7:0
ldnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency_1:0
_dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0
`dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f:0
Udnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalar:0
`dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge:0
Xdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/shape:0
Ydnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/shape:0�
Xdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/shape:0gdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2:0�
`dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge:0ednn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch:0�
_dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0_dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0�
`dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f:0`dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f:0�
Ydnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/shape:0gdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1:0�
Udnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalar:0gdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3:0
�
Fdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/cond_textFdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/pred_id:0Gdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_t:0 *�
Qdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/control_dependency:0
Fdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/pred_id:0
Gdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_t:0�
Gdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_t:0Gdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_t:0�
Fdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/pred_id:0Fdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/pred_id:0
�
Hdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/cond_text_1Fdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/pred_id:0Gdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_f:0*�
dnn/head/metrics/auc/Cast/x:0
/dnn/head/metrics/auc/assert_greater_equal/All:0
Ldnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/Switch:0
Ndnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/Switch_1:0
Ndnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/Switch_2:0
Ldnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/data_0:0
Ldnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/data_1:0
Ldnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/data_3:0
Sdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/control_dependency_1:0
Fdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/pred_id:0
Gdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_f:0
dnn/head/predictions/logistic:0�
Gdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_f:0Gdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_f:0q
dnn/head/predictions/logistic:0Ndnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/Switch_1:0�
Fdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/pred_id:0Fdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/pred_id:0o
dnn/head/metrics/auc/Cast/x:0Ndnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/Switch_2:0
/dnn/head/metrics/auc/assert_greater_equal/All:0Ldnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/Switch:0
�
Cdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/cond_textCdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/pred_id:0Ddnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_t:0 *�
Ndnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/control_dependency:0
Cdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/pred_id:0
Ddnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_t:0�
Cdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/pred_id:0Cdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/pred_id:0�
Ddnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_t:0Ddnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_t:0
�
Ednn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/cond_text_1Cdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/pred_id:0Ddnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_f:0*�
dnn/head/metrics/auc/Cast_1/x:0
,dnn/head/metrics/auc/assert_less_equal/All:0
Idnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/Switch:0
Kdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/Switch_1:0
Kdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/Switch_2:0
Idnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/data_0:0
Idnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/data_1:0
Idnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/data_3:0
Pdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/control_dependency_1:0
Cdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/pred_id:0
Ddnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_f:0
dnn/head/predictions/logistic:0n
dnn/head/metrics/auc/Cast_1/x:0Kdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/Switch_2:0n
dnn/head/predictions/logistic:0Kdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/Switch_1:0y
,dnn/head/metrics/auc/assert_less_equal/All:0Idnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/Switch:0�
Ddnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_f:0Ddnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_f:0�
Cdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/pred_id:0Cdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/pred_id:0
�	
Vdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/cond_textVdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_t:0 *�
Idnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalar:0
Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1:0
Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1:1
Vdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0
Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_t:0�
Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_t:0Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_t:0�
Idnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalar:0Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1:1�
Vdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0Vdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0
�_
Xdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/cond_text_1Vdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_f:0*�*
ndnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge:0
ndnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge:1
odnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:0
odnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1
qdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
qdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:1
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:0
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:2
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:0
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:0
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:0
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:0
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:0
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:0
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:0
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:0
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:0
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:0
{dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:0
ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:0
|dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch:0
~dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1:0
udnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0
pdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
qdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0
qdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0
Vdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0
Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_f:0
Kdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/rank:0
Ldnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/shape:0
Ldnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/rank:0
Mdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/shape:0�
Kdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/rank:0|dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch:0�
Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_f:0Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_f:0�
Mdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/shape:0�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0�
Vdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0Vdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0�
Ldnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/shape:0�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0�
Ldnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/rank:0~dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1:02�&
�&
pdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_textpdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0qdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0 *�#
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:0
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:2
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:0
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:0
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:0
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:0
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:0
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:0
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:0
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:0
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:0
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:0
{dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:0
ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:0
pdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
qdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0
Ldnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/shape:0
Mdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/shape:0�
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0�
Ldnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/shape:0�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1�
qdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0qdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0�
pdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0pdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0�
�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0�
Mdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/shape:0�dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:12�
�
rdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_text_1pdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0qdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0*�	
qdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
qdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:1
udnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0
pdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
qdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0�
udnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0qdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0�
qdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0qdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0�
pdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0pdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
�
Sdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/cond_textSdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0Tdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t:0 *�
^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency:0
Sdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0
Tdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t:0�
Tdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t:0Tdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t:0�
Sdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0Sdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0
�
Udnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/cond_text_1Sdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0Tdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f:0*�
Ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch:0
[dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1:0
[dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2:0
[dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3:0
Ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0:0
Ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1:0
Ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2:0
Ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4:0
Ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5:0
Ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7:0
`dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency_1:0
Sdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0
Tdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f:0
Idnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalar:0
Tdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge:0
Ldnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/shape:0
Mdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/shape:0�
Sdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0Sdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0�
Tdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f:0Tdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f:0�
Mdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/shape:0[dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1:0�
Idnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalar:0[dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3:0�
Ldnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/shape:0[dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2:0�
Tdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge:0Ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch:0
�
Wdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/cond_textWdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/pred_id:0Xdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_t:0 *�
bdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/control_dependency:0
Wdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/pred_id:0
Xdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_t:0�
Wdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/pred_id:0Wdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/pred_id:0�
Xdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_t:0Xdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_t:0
�
Ydnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/cond_text_1Wdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/pred_id:0Xdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_f:0*�
.dnn/head/metrics/auc_precision_recall/Cast/x:0
@dnn/head/metrics/auc_precision_recall/assert_greater_equal/All:0
]dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/Switch:0
_dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/Switch_1:0
_dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/Switch_2:0
]dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/data_0:0
]dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/data_1:0
]dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/data_3:0
ddnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/control_dependency_1:0
Wdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/pred_id:0
Xdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_f:0
dnn/head/predictions/logistic:0�
Xdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_f:0Xdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_f:0�
Wdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/pred_id:0Wdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/pred_id:0�
.dnn/head/metrics/auc_precision_recall/Cast/x:0_dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/Switch_2:0�
dnn/head/predictions/logistic:0_dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/Switch_1:0�
@dnn/head/metrics/auc_precision_recall/assert_greater_equal/All:0]dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/Switch:0
�
Tdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/cond_textTdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/pred_id:0Udnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_t:0 *�
_dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/control_dependency:0
Tdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/pred_id:0
Udnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_t:0�
Udnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_t:0Udnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_t:0�
Tdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/pred_id:0Tdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/pred_id:0
�
Vdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/cond_text_1Tdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/pred_id:0Udnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_f:0*�
0dnn/head/metrics/auc_precision_recall/Cast_1/x:0
=dnn/head/metrics/auc_precision_recall/assert_less_equal/All:0
Zdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/Switch:0
\dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/Switch_1:0
\dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/Switch_2:0
Zdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/data_0:0
Zdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/data_1:0
Zdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/data_3:0
adnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/control_dependency_1:0
Tdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/pred_id:0
Udnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_f:0
dnn/head/predictions/logistic:0�
0dnn/head/metrics/auc_precision_recall/Cast_1/x:0\dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/Switch_2:0�
Udnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_f:0Udnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_f:0
dnn/head/predictions/logistic:0\dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/Switch_1:0�
Tdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/pred_id:0Tdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/pred_id:0�
=dnn/head/metrics/auc_precision_recall/assert_less_equal/All:0Zdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/Switch:0
�
gdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/cond_textgdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_t:0 *�
Zdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalar:0
hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1:0
hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1:1
gdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0
hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_t:0�
Zdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalar:0hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1:1�
gdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0gdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0�
hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_t:0hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_t:0
�m
idnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/cond_text_1gdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_f:0*�1
dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge:0
dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge:1
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:0
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:1
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:0
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:2
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:0
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:0
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:0
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:0
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:0
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:0
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:0
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:0
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:0
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:0
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:0
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:0
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch:0
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1:0
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0
gdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0
hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_f:0
\dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/rank:0
]dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/shape:0
]dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/rank:0
^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/shape:0�
]dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/rank:0�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1:0�
gdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0gdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0�
^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/shape:0�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0�
\dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/rank:0�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch:0�
]dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/shape:0�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0�
hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_f:0hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_f:02�+
�+
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_text�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0 *�(
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:0
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:2
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:0
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:0
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:0
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:0
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:0
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:0
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:0
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:0
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:0
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:0
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:0
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:0
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0
]dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/shape:0
^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/shape:0�
^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/shape:0�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1�
]dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/shape:0�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:02�
�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_text_1�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0*�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:1
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0�
�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0�dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
�
ddnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/cond_textddnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0ednn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t:0 *�
odnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency:0
ddnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0
ednn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t:0�
ddnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0ddnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0�
ednn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t:0ednn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t:0
�
fdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/cond_text_1ddnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0ednn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f:0*�
jdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch:0
ldnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1:0
ldnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2:0
ldnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3:0
jdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0:0
jdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1:0
jdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2:0
jdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4:0
jdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5:0
jdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7:0
qdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency_1:0
ddnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0
ednn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f:0
Zdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalar:0
ednn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge:0
]dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/shape:0
^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/shape:0�
ddnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0ddnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0�
ednn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge:0jdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch:0�
ednn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f:0ednn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f:0�
^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/shape:0ldnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1:0�
]dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/shape:0ldnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2:0�
Zdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalar:0ldnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3:0"�
metric_variables�
�
#dnn/head/metrics/label/mean/total:0
#dnn/head/metrics/label/mean/count:0
%dnn/head/metrics/average_loss/total:0
%dnn/head/metrics/average_loss/count:0
!dnn/head/metrics/accuracy/total:0
!dnn/head/metrics/accuracy/count:0
1dnn/head/metrics/precision/true_positives/count:0
2dnn/head/metrics/precision/false_positives/count:0
.dnn/head/metrics/recall/true_positives/count:0
/dnn/head/metrics/recall/false_negatives/count:0
(dnn/head/metrics/prediction/mean/total:0
(dnn/head/metrics/prediction/mean/count:0
%dnn/head/metrics/auc/true_positives:0
&dnn/head/metrics/auc/false_negatives:0
%dnn/head/metrics/auc/true_negatives:0
&dnn/head/metrics/auc/false_positives:0
6dnn/head/metrics/auc_precision_recall/true_positives:0
7dnn/head/metrics/auc_precision_recall/false_negatives:0
6dnn/head/metrics/auc_precision_recall/true_negatives:0
7dnn/head/metrics/auc_precision_recall/false_positives:0
mean/total:0
mean/count:0"�"
local_variables�"�"
�
#dnn/head/metrics/label/mean/total:0(dnn/head/metrics/label/mean/total/Assign(dnn/head/metrics/label/mean/total/read:025dnn/head/metrics/label/mean/total/Initializer/zeros:0
�
#dnn/head/metrics/label/mean/count:0(dnn/head/metrics/label/mean/count/Assign(dnn/head/metrics/label/mean/count/read:025dnn/head/metrics/label/mean/count/Initializer/zeros:0
�
%dnn/head/metrics/average_loss/total:0*dnn/head/metrics/average_loss/total/Assign*dnn/head/metrics/average_loss/total/read:027dnn/head/metrics/average_loss/total/Initializer/zeros:0
�
%dnn/head/metrics/average_loss/count:0*dnn/head/metrics/average_loss/count/Assign*dnn/head/metrics/average_loss/count/read:027dnn/head/metrics/average_loss/count/Initializer/zeros:0
�
!dnn/head/metrics/accuracy/total:0&dnn/head/metrics/accuracy/total/Assign&dnn/head/metrics/accuracy/total/read:023dnn/head/metrics/accuracy/total/Initializer/zeros:0
�
!dnn/head/metrics/accuracy/count:0&dnn/head/metrics/accuracy/count/Assign&dnn/head/metrics/accuracy/count/read:023dnn/head/metrics/accuracy/count/Initializer/zeros:0
�
1dnn/head/metrics/precision/true_positives/count:06dnn/head/metrics/precision/true_positives/count/Assign6dnn/head/metrics/precision/true_positives/count/read:02Cdnn/head/metrics/precision/true_positives/count/Initializer/zeros:0
�
2dnn/head/metrics/precision/false_positives/count:07dnn/head/metrics/precision/false_positives/count/Assign7dnn/head/metrics/precision/false_positives/count/read:02Ddnn/head/metrics/precision/false_positives/count/Initializer/zeros:0
�
.dnn/head/metrics/recall/true_positives/count:03dnn/head/metrics/recall/true_positives/count/Assign3dnn/head/metrics/recall/true_positives/count/read:02@dnn/head/metrics/recall/true_positives/count/Initializer/zeros:0
�
/dnn/head/metrics/recall/false_negatives/count:04dnn/head/metrics/recall/false_negatives/count/Assign4dnn/head/metrics/recall/false_negatives/count/read:02Adnn/head/metrics/recall/false_negatives/count/Initializer/zeros:0
�
(dnn/head/metrics/prediction/mean/total:0-dnn/head/metrics/prediction/mean/total/Assign-dnn/head/metrics/prediction/mean/total/read:02:dnn/head/metrics/prediction/mean/total/Initializer/zeros:0
�
(dnn/head/metrics/prediction/mean/count:0-dnn/head/metrics/prediction/mean/count/Assign-dnn/head/metrics/prediction/mean/count/read:02:dnn/head/metrics/prediction/mean/count/Initializer/zeros:0
�
%dnn/head/metrics/auc/true_positives:0*dnn/head/metrics/auc/true_positives/Assign*dnn/head/metrics/auc/true_positives/read:027dnn/head/metrics/auc/true_positives/Initializer/zeros:0
�
&dnn/head/metrics/auc/false_negatives:0+dnn/head/metrics/auc/false_negatives/Assign+dnn/head/metrics/auc/false_negatives/read:028dnn/head/metrics/auc/false_negatives/Initializer/zeros:0
�
%dnn/head/metrics/auc/true_negatives:0*dnn/head/metrics/auc/true_negatives/Assign*dnn/head/metrics/auc/true_negatives/read:027dnn/head/metrics/auc/true_negatives/Initializer/zeros:0
�
&dnn/head/metrics/auc/false_positives:0+dnn/head/metrics/auc/false_positives/Assign+dnn/head/metrics/auc/false_positives/read:028dnn/head/metrics/auc/false_positives/Initializer/zeros:0
�
6dnn/head/metrics/auc_precision_recall/true_positives:0;dnn/head/metrics/auc_precision_recall/true_positives/Assign;dnn/head/metrics/auc_precision_recall/true_positives/read:02Hdnn/head/metrics/auc_precision_recall/true_positives/Initializer/zeros:0
�
7dnn/head/metrics/auc_precision_recall/false_negatives:0<dnn/head/metrics/auc_precision_recall/false_negatives/Assign<dnn/head/metrics/auc_precision_recall/false_negatives/read:02Idnn/head/metrics/auc_precision_recall/false_negatives/Initializer/zeros:0
�
6dnn/head/metrics/auc_precision_recall/true_negatives:0;dnn/head/metrics/auc_precision_recall/true_negatives/Assign;dnn/head/metrics/auc_precision_recall/true_negatives/read:02Hdnn/head/metrics/auc_precision_recall/true_negatives/Initializer/zeros:0
�
7dnn/head/metrics/auc_precision_recall/false_positives:0<dnn/head/metrics/auc_precision_recall/false_positives/Assign<dnn/head/metrics/auc_precision_recall/false_positives/read:02Idnn/head/metrics/auc_precision_recall/false_positives/Initializer/zeros:0
T
mean/total:0mean/total/Assignmean/total/read:02mean/total/Initializer/zeros:0
T
mean/count:0mean/count/Assignmean/count/read:02mean/count/Initializer/zeros:0
P
eval_step:0eval_step/Assigneval_step/read:02eval_step/Initializer/zeros:0"!
local_init_op

group_deps_2"�

	variables�
�

X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0
�
!dnn/hiddenlayer_0/kernel/part_0:0&dnn/hiddenlayer_0/kernel/part_0/Assign&dnn/hiddenlayer_0/kernel/part_0/read:0"*
dnn/hiddenlayer_0/kernel� �  "� �2<dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform:0
�
dnn/hiddenlayer_0/bias/part_0:0$dnn/hiddenlayer_0/bias/part_0/Assign$dnn/hiddenlayer_0/bias/part_0/read:0"#
dnn/hiddenlayer_0/bias� "�21dnn/hiddenlayer_0/bias/part_0/Initializer/zeros:0
�
!dnn/hiddenlayer_1/kernel/part_0:0&dnn/hiddenlayer_1/kernel/part_0/Assign&dnn/hiddenlayer_1/kernel/part_0/read:0"(
dnn/hiddenlayer_1/kernel�   "� 2<dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform:0
�
dnn/hiddenlayer_1/bias/part_0:0$dnn/hiddenlayer_1/bias/part_0/Assign$dnn/hiddenlayer_1/bias/part_0/read:0"!
dnn/hiddenlayer_1/bias  " 21dnn/hiddenlayer_1/bias/part_0/Initializer/zeros:0
�
dnn/logits/kernel/part_0:0dnn/logits/kernel/part_0/Assigndnn/logits/kernel/part_0/read:0"
dnn/logits/kernel   " 25dnn/logits/kernel/part_0/Initializer/random_uniform:0
�
dnn/logits/bias/part_0:0dnn/logits/bias/part_0/Assigndnn/logits/bias/part_0/read:0"
dnn/logits/bias "2*dnn/logits/bias/part_0/Initializer/zeros:0"
ready_op


concat:0"J
savers@>
<
save/Const:0save/Identity:0save/restore_all (5 @F8"�
queue_runners��
�
enqueue_input/fifo_queue$enqueue_input/fifo_queue_EnqueueManyenqueue_input/fifo_queue_Close" enqueue_input/fifo_queue_Close_1*"*
losses 

dnn/head/weighted_loss/Sum:0"k
global_step\Z
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0"&

summary_op

Merge/MergeSummary:0n����       �7So	<�0:��A*�

accuracy!?

accuracy_baseline!?


auc���>

auc_precision_recall�B?

average_loss-b1?


label/mean!?

loss-b1?

	precision!?

prediction/mean$?

recall  �?�G;��       �7So	֍XB��A*�

accuracy��>

accuracy_baseline!?


auc���>

auc_precision_recall�B?

average_loss6�8?


label/mean!?

loss6�8?

	precision    

prediction/mean��>

recall    *4��       �7So	>�6E��A*�

accuracy!?

accuracy_baseline!?


auc���>

auc_precision_recall�B?

average_loss^1?


label/mean!?

loss^1?

	precision!?

prediction/mean��?

recall  �?]��       �7So	n�b��A*�

accuracy��>

accuracy_baseline!?


auc���>

auc_precision_recall�B?

average_loss��7?


label/mean!?

loss��7?

	precision    

prediction/mean���>

recall    �-�y