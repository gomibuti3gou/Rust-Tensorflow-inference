с
эО
B
AssignVariableOp
resource
value"dtype"
dtypetype
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
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
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
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
О
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.6.02v2.6.0-rc2-32-g919f693420e8йч

cnn_7/conv2d_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namecnn_7/conv2d_21/kernel

*cnn_7/conv2d_21/kernel/Read/ReadVariableOpReadVariableOpcnn_7/conv2d_21/kernel*&
_output_shapes
: *
dtype0

cnn_7/conv2d_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_namecnn_7/conv2d_21/bias
y
(cnn_7/conv2d_21/bias/Read/ReadVariableOpReadVariableOpcnn_7/conv2d_21/bias*
_output_shapes
: *
dtype0

cnn_7/conv2d_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_namecnn_7/conv2d_22/kernel

*cnn_7/conv2d_22/kernel/Read/ReadVariableOpReadVariableOpcnn_7/conv2d_22/kernel*&
_output_shapes
: @*
dtype0

cnn_7/conv2d_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_namecnn_7/conv2d_22/bias
y
(cnn_7/conv2d_22/bias/Read/ReadVariableOpReadVariableOpcnn_7/conv2d_22/bias*
_output_shapes
:@*
dtype0

cnn_7/conv2d_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_namecnn_7/conv2d_23/kernel

*cnn_7/conv2d_23/kernel/Read/ReadVariableOpReadVariableOpcnn_7/conv2d_23/kernel*&
_output_shapes
:@@*
dtype0

cnn_7/conv2d_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_namecnn_7/conv2d_23/bias
y
(cnn_7/conv2d_23/bias/Read/ReadVariableOpReadVariableOpcnn_7/conv2d_23/bias*
_output_shapes
:@*
dtype0

cnn_7/dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Р@*&
shared_namecnn_7/dense_14/kernel

)cnn_7/dense_14/kernel/Read/ReadVariableOpReadVariableOpcnn_7/dense_14/kernel*
_output_shapes
:	Р@*
dtype0
~
cnn_7/dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_namecnn_7/dense_14/bias
w
'cnn_7/dense_14/bias/Read/ReadVariableOpReadVariableOpcnn_7/dense_14/bias*
_output_shapes
:@*
dtype0

cnn_7/dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@
*&
shared_namecnn_7/dense_15/kernel

)cnn_7/dense_15/kernel/Read/ReadVariableOpReadVariableOpcnn_7/dense_15/kernel*
_output_shapes

:@
*
dtype0
~
cnn_7/dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_namecnn_7/dense_15/bias
w
'cnn_7/dense_15/bias/Read/ReadVariableOpReadVariableOpcnn_7/dense_15/bias*
_output_shapes
:
*
dtype0

NoOpNoOp
у"
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*"
value"B" B"

layer-0
layer_with_weights-0
layer-1
	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
И
	conv1
	maxpooling1
	
conv2
maxpooling2
	conv3
flatten
	dens1
	dens2
	variables
regularization_losses
trainable_variables
	keras_api
F
0
1
2
3
4
5
6
7
8
9
 
F
0
1
2
3
4
5
6
7
8
9
­
layer_regularization_losses
	variables
layer_metrics
regularization_losses
 metrics

!layers
trainable_variables
"non_trainable_variables
 
h

kernel
bias
#	variables
$regularization_losses
%trainable_variables
&	keras_api
R
'	variables
(regularization_losses
)trainable_variables
*	keras_api
h

kernel
bias
+	variables
,regularization_losses
-trainable_variables
.	keras_api
R
/	variables
0regularization_losses
1trainable_variables
2	keras_api
h

kernel
bias
3	variables
4regularization_losses
5trainable_variables
6	keras_api
R
7	variables
8regularization_losses
9trainable_variables
:	keras_api
h

kernel
bias
;	variables
<regularization_losses
=trainable_variables
>	keras_api
h

kernel
bias
?	variables
@regularization_losses
Atrainable_variables
B	keras_api
F
0
1
2
3
4
5
6
7
8
9
 
F
0
1
2
3
4
5
6
7
8
9
­
Clayer_regularization_losses
	variables
Dlayer_metrics
regularization_losses
Emetrics

Flayers
trainable_variables
Gnon_trainable_variables
RP
VARIABLE_VALUEcnn_7/conv2d_21/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEcnn_7/conv2d_21/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcnn_7/conv2d_22/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEcnn_7/conv2d_22/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcnn_7/conv2d_23/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEcnn_7/conv2d_23/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcnn_7/dense_14/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcnn_7/dense_14/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcnn_7/dense_15/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcnn_7/dense_15/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

0
1
 

0
1
 

0
1
­
Hlayer_regularization_losses

Ilayers
#	variables
$regularization_losses
Jmetrics
Klayer_metrics
%trainable_variables
Lnon_trainable_variables
 
 
 
­
Mlayer_regularization_losses

Nlayers
'	variables
(regularization_losses
Ometrics
Player_metrics
)trainable_variables
Qnon_trainable_variables

0
1
 

0
1
­
Rlayer_regularization_losses

Slayers
+	variables
,regularization_losses
Tmetrics
Ulayer_metrics
-trainable_variables
Vnon_trainable_variables
 
 
 
­
Wlayer_regularization_losses

Xlayers
/	variables
0regularization_losses
Ymetrics
Zlayer_metrics
1trainable_variables
[non_trainable_variables

0
1
 

0
1
­
\layer_regularization_losses

]layers
3	variables
4regularization_losses
^metrics
_layer_metrics
5trainable_variables
`non_trainable_variables
 
 
 
­
alayer_regularization_losses

blayers
7	variables
8regularization_losses
cmetrics
dlayer_metrics
9trainable_variables
enon_trainable_variables

0
1
 

0
1
­
flayer_regularization_losses

glayers
;	variables
<regularization_losses
hmetrics
ilayer_metrics
=trainable_variables
jnon_trainable_variables

0
1
 

0
1
­
klayer_regularization_losses

llayers
?	variables
@regularization_losses
mmetrics
nlayer_metrics
Atrainable_variables
onon_trainable_variables
 
 
 
8
0
	1

2
3
4
5
6
7
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

serving_default_inputPlaceholder*/
_output_shapes
:џџџџџџџџџ*
dtype0*$
shape:џџџџџџџџџ
І
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputcnn_7/conv2d_21/kernelcnn_7/conv2d_21/biascnn_7/conv2d_22/kernelcnn_7/conv2d_22/biascnn_7/conv2d_23/kernelcnn_7/conv2d_23/biascnn_7/dense_14/kernelcnn_7/dense_14/biascnn_7/dense_15/kernelcnn_7/dense_15/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_206031
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Я
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*cnn_7/conv2d_21/kernel/Read/ReadVariableOp(cnn_7/conv2d_21/bias/Read/ReadVariableOp*cnn_7/conv2d_22/kernel/Read/ReadVariableOp(cnn_7/conv2d_22/bias/Read/ReadVariableOp*cnn_7/conv2d_23/kernel/Read/ReadVariableOp(cnn_7/conv2d_23/bias/Read/ReadVariableOp)cnn_7/dense_14/kernel/Read/ReadVariableOp'cnn_7/dense_14/bias/Read/ReadVariableOp)cnn_7/dense_15/kernel/Read/ReadVariableOp'cnn_7/dense_15/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_206507

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamecnn_7/conv2d_21/kernelcnn_7/conv2d_21/biascnn_7/conv2d_22/kernelcnn_7/conv2d_22/biascnn_7/conv2d_23/kernelcnn_7/conv2d_23/biascnn_7/dense_14/kernelcnn_7/dense_14/biascnn_7/dense_15/kernelcnn_7/dense_15/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_206547 
Ќ
h
L__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_206338

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ч


)__inference_model_14_layer_call_fn_205854	
input!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:	Р@
	unknown_6:@
	unknown_7:@

	unknown_8:

identityЂStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_model_14_layer_call_and_return_conditional_losses_2058312
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:џџџџџџџџџ

_user_specified_nameinput
ь
M
1__inference_max_pooling2d_15_layer_call_fn_206373

inputs
identityв
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_2055942
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
э
Ш
D__inference_model_14_layer_call_and_return_conditional_losses_205831

inputs&
cnn_7_205809: 
cnn_7_205811: &
cnn_7_205813: @
cnn_7_205815:@&
cnn_7_205817:@@
cnn_7_205819:@
cnn_7_205821:	Р@
cnn_7_205823:@
cnn_7_205825:@

cnn_7_205827:

identityЂcnn_7/StatefulPartitionedCall
cnn_7/StatefulPartitionedCallStatefulPartitionedCallinputscnn_7_205809cnn_7_205811cnn_7_205813cnn_7_205815cnn_7_205817cnn_7_205819cnn_7_205821cnn_7_205823cnn_7_205825cnn_7_205827*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_cnn_7_layer_call_and_return_conditional_losses_2056562
cnn_7/StatefulPartitionedCall
IdentityIdentity&cnn_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identityn
NoOpNoOp^cnn_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ: : : : : : : : : : 2>
cnn_7/StatefulPartitionedCallcnn_7/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

ѕ
D__inference_dense_15_layer_call_and_return_conditional_losses_205649

inputs0
matmul_readvariableop_resource:@
-
biasadd_readvariableop_resource:

identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ч


&__inference_cnn_7_layer_call_fn_206192
input_1!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:	Р@
	unknown_6:@
	unknown_7:@

	unknown_8:

identityЂStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_cnn_7_layer_call_and_return_conditional_losses_2056562
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
/
№
"__inference__traced_restore_206547
file_prefixA
'assignvariableop_cnn_7_conv2d_21_kernel: 5
'assignvariableop_1_cnn_7_conv2d_21_bias: C
)assignvariableop_2_cnn_7_conv2d_22_kernel: @5
'assignvariableop_3_cnn_7_conv2d_22_bias:@C
)assignvariableop_4_cnn_7_conv2d_23_kernel:@@5
'assignvariableop_5_cnn_7_conv2d_23_bias:@;
(assignvariableop_6_cnn_7_dense_14_kernel:	Р@4
&assignvariableop_7_cnn_7_dense_14_bias:@:
(assignvariableop_8_cnn_7_dense_15_kernel:@
4
&assignvariableop_9_cnn_7_dense_15_bias:

identity_11ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_2ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9З
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*У
valueЙBЖB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЄ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
RestoreV2/shape_and_slicesт
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityІ
AssignVariableOpAssignVariableOp'assignvariableop_cnn_7_conv2d_21_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ќ
AssignVariableOp_1AssignVariableOp'assignvariableop_1_cnn_7_conv2d_21_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ў
AssignVariableOp_2AssignVariableOp)assignvariableop_2_cnn_7_conv2d_22_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ќ
AssignVariableOp_3AssignVariableOp'assignvariableop_3_cnn_7_conv2d_22_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Ў
AssignVariableOp_4AssignVariableOp)assignvariableop_4_cnn_7_conv2d_23_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ќ
AssignVariableOp_5AssignVariableOp'assignvariableop_5_cnn_7_conv2d_23_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6­
AssignVariableOp_6AssignVariableOp(assignvariableop_6_cnn_7_dense_14_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ћ
AssignVariableOp_7AssignVariableOp&assignvariableop_7_cnn_7_dense_14_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8­
AssignVariableOp_8AssignVariableOp(assignvariableop_8_cnn_7_dense_15_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ћ
AssignVariableOp_9AssignVariableOp&assignvariableop_9_cnn_7_dense_15_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpК
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_10f
Identity_11IdentityIdentity_10:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_11Ђ
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_11Identity_11:output:0*)
_input_shapes
: : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
є

)__inference_dense_14_layer_call_fn_206423

inputs
unknown:	Р@
	unknown_0:@
identityЂStatefulPartitionedCallє
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_2056322
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџР: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs
ё

)__inference_dense_15_layer_call_fn_206443

inputs
unknown:@

	unknown_0:

identityЂStatefulPartitionedCallє
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_2056492
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ъ
Ч
D__inference_model_14_layer_call_and_return_conditional_losses_206004	
input&
cnn_7_205982: 
cnn_7_205984: &
cnn_7_205986: @
cnn_7_205988:@&
cnn_7_205990:@@
cnn_7_205992:@
cnn_7_205994:	Р@
cnn_7_205996:@
cnn_7_205998:@

cnn_7_206000:

identityЂcnn_7/StatefulPartitionedCall
cnn_7/StatefulPartitionedCallStatefulPartitionedCallinputcnn_7_205982cnn_7_205984cnn_7_205986cnn_7_205988cnn_7_205990cnn_7_205992cnn_7_205994cnn_7_205996cnn_7_205998cnn_7_206000*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_cnn_7_layer_call_and_return_conditional_losses_2056562
cnn_7/StatefulPartitionedCall
IdentityIdentity&cnn_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identityn
NoOpNoOp^cnn_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ: : : : : : : : : : 2>
cnn_7/StatefulPartitionedCallcnn_7/StatefulPartitionedCall:V R
/
_output_shapes
:џџџџџџџџџ

_user_specified_nameinput
П


$__inference_signature_wrapper_206031	
input!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:	Р@
	unknown_6:@
	unknown_7:@

	unknown_8:

identityЂStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_2054992
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:џџџџџџџџџ

_user_specified_nameinput
й
M
1__inference_max_pooling2d_15_layer_call_fn_206368

inputs
identityэ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_2055302
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ђJ


!__inference__wrapped_model_205499	
inputQ
7model_14_cnn_7_conv2d_21_conv2d_readvariableop_resource: F
8model_14_cnn_7_conv2d_21_biasadd_readvariableop_resource: Q
7model_14_cnn_7_conv2d_22_conv2d_readvariableop_resource: @F
8model_14_cnn_7_conv2d_22_biasadd_readvariableop_resource:@Q
7model_14_cnn_7_conv2d_23_conv2d_readvariableop_resource:@@F
8model_14_cnn_7_conv2d_23_biasadd_readvariableop_resource:@I
6model_14_cnn_7_dense_14_matmul_readvariableop_resource:	Р@E
7model_14_cnn_7_dense_14_biasadd_readvariableop_resource:@H
6model_14_cnn_7_dense_15_matmul_readvariableop_resource:@
E
7model_14_cnn_7_dense_15_biasadd_readvariableop_resource:

identityЂ/model_14/cnn_7/conv2d_21/BiasAdd/ReadVariableOpЂ.model_14/cnn_7/conv2d_21/Conv2D/ReadVariableOpЂ/model_14/cnn_7/conv2d_22/BiasAdd/ReadVariableOpЂ.model_14/cnn_7/conv2d_22/Conv2D/ReadVariableOpЂ/model_14/cnn_7/conv2d_23/BiasAdd/ReadVariableOpЂ.model_14/cnn_7/conv2d_23/Conv2D/ReadVariableOpЂ.model_14/cnn_7/dense_14/BiasAdd/ReadVariableOpЂ-model_14/cnn_7/dense_14/MatMul/ReadVariableOpЂ.model_14/cnn_7/dense_15/BiasAdd/ReadVariableOpЂ-model_14/cnn_7/dense_15/MatMul/ReadVariableOpр
.model_14/cnn_7/conv2d_21/Conv2D/ReadVariableOpReadVariableOp7model_14_cnn_7_conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype020
.model_14/cnn_7/conv2d_21/Conv2D/ReadVariableOpю
model_14/cnn_7/conv2d_21/Conv2DConv2Dinput6model_14/cnn_7/conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2!
model_14/cnn_7/conv2d_21/Conv2Dз
/model_14/cnn_7/conv2d_21/BiasAdd/ReadVariableOpReadVariableOp8model_14_cnn_7_conv2d_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/model_14/cnn_7/conv2d_21/BiasAdd/ReadVariableOpь
 model_14/cnn_7/conv2d_21/BiasAddBiasAdd(model_14/cnn_7/conv2d_21/Conv2D:output:07model_14/cnn_7/conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2"
 model_14/cnn_7/conv2d_21/BiasAddЋ
model_14/cnn_7/conv2d_21/ReluRelu)model_14/cnn_7/conv2d_21/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
model_14/cnn_7/conv2d_21/Reluї
'model_14/cnn_7/max_pooling2d_14/MaxPoolMaxPool+model_14/cnn_7/conv2d_21/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingVALID*
strides
2)
'model_14/cnn_7/max_pooling2d_14/MaxPoolр
.model_14/cnn_7/conv2d_22/Conv2D/ReadVariableOpReadVariableOp7model_14_cnn_7_conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype020
.model_14/cnn_7/conv2d_22/Conv2D/ReadVariableOp
model_14/cnn_7/conv2d_22/Conv2DConv2D0model_14/cnn_7/max_pooling2d_14/MaxPool:output:06model_14/cnn_7/conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2!
model_14/cnn_7/conv2d_22/Conv2Dз
/model_14/cnn_7/conv2d_22/BiasAdd/ReadVariableOpReadVariableOp8model_14_cnn_7_conv2d_22_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/model_14/cnn_7/conv2d_22/BiasAdd/ReadVariableOpь
 model_14/cnn_7/conv2d_22/BiasAddBiasAdd(model_14/cnn_7/conv2d_22/Conv2D:output:07model_14/cnn_7/conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2"
 model_14/cnn_7/conv2d_22/BiasAddЋ
model_14/cnn_7/conv2d_22/ReluRelu)model_14/cnn_7/conv2d_22/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
model_14/cnn_7/conv2d_22/Reluї
'model_14/cnn_7/max_pooling2d_15/MaxPoolMaxPool+model_14/cnn_7/conv2d_22/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides
2)
'model_14/cnn_7/max_pooling2d_15/MaxPoolр
.model_14/cnn_7/conv2d_23/Conv2D/ReadVariableOpReadVariableOp7model_14_cnn_7_conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype020
.model_14/cnn_7/conv2d_23/Conv2D/ReadVariableOp
model_14/cnn_7/conv2d_23/Conv2DConv2D0model_14/cnn_7/max_pooling2d_15/MaxPool:output:06model_14/cnn_7/conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2!
model_14/cnn_7/conv2d_23/Conv2Dз
/model_14/cnn_7/conv2d_23/BiasAdd/ReadVariableOpReadVariableOp8model_14_cnn_7_conv2d_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/model_14/cnn_7/conv2d_23/BiasAdd/ReadVariableOpь
 model_14/cnn_7/conv2d_23/BiasAddBiasAdd(model_14/cnn_7/conv2d_23/Conv2D:output:07model_14/cnn_7/conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2"
 model_14/cnn_7/conv2d_23/BiasAddЋ
model_14/cnn_7/conv2d_23/ReluRelu)model_14/cnn_7/conv2d_23/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
model_14/cnn_7/conv2d_23/Relu
model_14/cnn_7/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@  2 
model_14/cnn_7/flatten_7/Constи
 model_14/cnn_7/flatten_7/ReshapeReshape+model_14/cnn_7/conv2d_23/Relu:activations:0'model_14/cnn_7/flatten_7/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2"
 model_14/cnn_7/flatten_7/Reshapeж
-model_14/cnn_7/dense_14/MatMul/ReadVariableOpReadVariableOp6model_14_cnn_7_dense_14_matmul_readvariableop_resource*
_output_shapes
:	Р@*
dtype02/
-model_14/cnn_7/dense_14/MatMul/ReadVariableOpо
model_14/cnn_7/dense_14/MatMulMatMul)model_14/cnn_7/flatten_7/Reshape:output:05model_14/cnn_7/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2 
model_14/cnn_7/dense_14/MatMulд
.model_14/cnn_7/dense_14/BiasAdd/ReadVariableOpReadVariableOp7model_14_cnn_7_dense_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.model_14/cnn_7/dense_14/BiasAdd/ReadVariableOpс
model_14/cnn_7/dense_14/BiasAddBiasAdd(model_14/cnn_7/dense_14/MatMul:product:06model_14/cnn_7/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2!
model_14/cnn_7/dense_14/BiasAdd 
model_14/cnn_7/dense_14/ReluRelu(model_14/cnn_7/dense_14/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
model_14/cnn_7/dense_14/Reluе
-model_14/cnn_7/dense_15/MatMul/ReadVariableOpReadVariableOp6model_14_cnn_7_dense_15_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype02/
-model_14/cnn_7/dense_15/MatMul/ReadVariableOpп
model_14/cnn_7/dense_15/MatMulMatMul*model_14/cnn_7/dense_14/Relu:activations:05model_14/cnn_7/dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2 
model_14/cnn_7/dense_15/MatMulд
.model_14/cnn_7/dense_15/BiasAdd/ReadVariableOpReadVariableOp7model_14_cnn_7_dense_15_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype020
.model_14/cnn_7/dense_15/BiasAdd/ReadVariableOpс
model_14/cnn_7/dense_15/BiasAddBiasAdd(model_14/cnn_7/dense_15/MatMul:product:06model_14/cnn_7/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2!
model_14/cnn_7/dense_15/BiasAddЉ
model_14/cnn_7/dense_15/SoftmaxSoftmax(model_14/cnn_7/dense_15/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
2!
model_14/cnn_7/dense_15/Softmax
IdentityIdentity)model_14/cnn_7/dense_15/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
2

IdentityЙ
NoOpNoOp0^model_14/cnn_7/conv2d_21/BiasAdd/ReadVariableOp/^model_14/cnn_7/conv2d_21/Conv2D/ReadVariableOp0^model_14/cnn_7/conv2d_22/BiasAdd/ReadVariableOp/^model_14/cnn_7/conv2d_22/Conv2D/ReadVariableOp0^model_14/cnn_7/conv2d_23/BiasAdd/ReadVariableOp/^model_14/cnn_7/conv2d_23/Conv2D/ReadVariableOp/^model_14/cnn_7/dense_14/BiasAdd/ReadVariableOp.^model_14/cnn_7/dense_14/MatMul/ReadVariableOp/^model_14/cnn_7/dense_15/BiasAdd/ReadVariableOp.^model_14/cnn_7/dense_15/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ: : : : : : : : : : 2b
/model_14/cnn_7/conv2d_21/BiasAdd/ReadVariableOp/model_14/cnn_7/conv2d_21/BiasAdd/ReadVariableOp2`
.model_14/cnn_7/conv2d_21/Conv2D/ReadVariableOp.model_14/cnn_7/conv2d_21/Conv2D/ReadVariableOp2b
/model_14/cnn_7/conv2d_22/BiasAdd/ReadVariableOp/model_14/cnn_7/conv2d_22/BiasAdd/ReadVariableOp2`
.model_14/cnn_7/conv2d_22/Conv2D/ReadVariableOp.model_14/cnn_7/conv2d_22/Conv2D/ReadVariableOp2b
/model_14/cnn_7/conv2d_23/BiasAdd/ReadVariableOp/model_14/cnn_7/conv2d_23/BiasAdd/ReadVariableOp2`
.model_14/cnn_7/conv2d_23/Conv2D/ReadVariableOp.model_14/cnn_7/conv2d_23/Conv2D/ReadVariableOp2`
.model_14/cnn_7/dense_14/BiasAdd/ReadVariableOp.model_14/cnn_7/dense_14/BiasAdd/ReadVariableOp2^
-model_14/cnn_7/dense_14/MatMul/ReadVariableOp-model_14/cnn_7/dense_14/MatMul/ReadVariableOp2`
.model_14/cnn_7/dense_15/BiasAdd/ReadVariableOp.model_14/cnn_7/dense_15/BiasAdd/ReadVariableOp2^
-model_14/cnn_7/dense_15/MatMul/ReadVariableOp-model_14/cnn_7/dense_15/MatMul/ReadVariableOp:V R
/
_output_shapes
:џџџџџџџџџ

_user_specified_nameinput
е

џ
&__inference_cnn_7_layer_call_fn_206217
x!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:	Р@
	unknown_6:@
	unknown_7:@

	unknown_8:

identityЂStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_cnn_7_layer_call_and_return_conditional_losses_2056562
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:џџџџџџџџџ

_user_specified_namex
ы@
	
D__inference_model_14_layer_call_and_return_conditional_losses_206124

inputsH
.cnn_7_conv2d_21_conv2d_readvariableop_resource: =
/cnn_7_conv2d_21_biasadd_readvariableop_resource: H
.cnn_7_conv2d_22_conv2d_readvariableop_resource: @=
/cnn_7_conv2d_22_biasadd_readvariableop_resource:@H
.cnn_7_conv2d_23_conv2d_readvariableop_resource:@@=
/cnn_7_conv2d_23_biasadd_readvariableop_resource:@@
-cnn_7_dense_14_matmul_readvariableop_resource:	Р@<
.cnn_7_dense_14_biasadd_readvariableop_resource:@?
-cnn_7_dense_15_matmul_readvariableop_resource:@
<
.cnn_7_dense_15_biasadd_readvariableop_resource:

identityЂ&cnn_7/conv2d_21/BiasAdd/ReadVariableOpЂ%cnn_7/conv2d_21/Conv2D/ReadVariableOpЂ&cnn_7/conv2d_22/BiasAdd/ReadVariableOpЂ%cnn_7/conv2d_22/Conv2D/ReadVariableOpЂ&cnn_7/conv2d_23/BiasAdd/ReadVariableOpЂ%cnn_7/conv2d_23/Conv2D/ReadVariableOpЂ%cnn_7/dense_14/BiasAdd/ReadVariableOpЂ$cnn_7/dense_14/MatMul/ReadVariableOpЂ%cnn_7/dense_15/BiasAdd/ReadVariableOpЂ$cnn_7/dense_15/MatMul/ReadVariableOpХ
%cnn_7/conv2d_21/Conv2D/ReadVariableOpReadVariableOp.cnn_7_conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02'
%cnn_7/conv2d_21/Conv2D/ReadVariableOpд
cnn_7/conv2d_21/Conv2DConv2Dinputs-cnn_7/conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2
cnn_7/conv2d_21/Conv2DМ
&cnn_7/conv2d_21/BiasAdd/ReadVariableOpReadVariableOp/cnn_7_conv2d_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&cnn_7/conv2d_21/BiasAdd/ReadVariableOpШ
cnn_7/conv2d_21/BiasAddBiasAddcnn_7/conv2d_21/Conv2D:output:0.cnn_7/conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
cnn_7/conv2d_21/BiasAdd
cnn_7/conv2d_21/ReluRelu cnn_7/conv2d_21/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
cnn_7/conv2d_21/Reluм
cnn_7/max_pooling2d_14/MaxPoolMaxPool"cnn_7/conv2d_21/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingVALID*
strides
2 
cnn_7/max_pooling2d_14/MaxPoolХ
%cnn_7/conv2d_22/Conv2D/ReadVariableOpReadVariableOp.cnn_7_conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02'
%cnn_7/conv2d_22/Conv2D/ReadVariableOpѕ
cnn_7/conv2d_22/Conv2DConv2D'cnn_7/max_pooling2d_14/MaxPool:output:0-cnn_7/conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2
cnn_7/conv2d_22/Conv2DМ
&cnn_7/conv2d_22/BiasAdd/ReadVariableOpReadVariableOp/cnn_7_conv2d_22_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&cnn_7/conv2d_22/BiasAdd/ReadVariableOpШ
cnn_7/conv2d_22/BiasAddBiasAddcnn_7/conv2d_22/Conv2D:output:0.cnn_7/conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
cnn_7/conv2d_22/BiasAdd
cnn_7/conv2d_22/ReluRelu cnn_7/conv2d_22/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
cnn_7/conv2d_22/Reluм
cnn_7/max_pooling2d_15/MaxPoolMaxPool"cnn_7/conv2d_22/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides
2 
cnn_7/max_pooling2d_15/MaxPoolХ
%cnn_7/conv2d_23/Conv2D/ReadVariableOpReadVariableOp.cnn_7_conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02'
%cnn_7/conv2d_23/Conv2D/ReadVariableOpѕ
cnn_7/conv2d_23/Conv2DConv2D'cnn_7/max_pooling2d_15/MaxPool:output:0-cnn_7/conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2
cnn_7/conv2d_23/Conv2DМ
&cnn_7/conv2d_23/BiasAdd/ReadVariableOpReadVariableOp/cnn_7_conv2d_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&cnn_7/conv2d_23/BiasAdd/ReadVariableOpШ
cnn_7/conv2d_23/BiasAddBiasAddcnn_7/conv2d_23/Conv2D:output:0.cnn_7/conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
cnn_7/conv2d_23/BiasAdd
cnn_7/conv2d_23/ReluRelu cnn_7/conv2d_23/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
cnn_7/conv2d_23/Relu
cnn_7/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@  2
cnn_7/flatten_7/ConstД
cnn_7/flatten_7/ReshapeReshape"cnn_7/conv2d_23/Relu:activations:0cnn_7/flatten_7/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2
cnn_7/flatten_7/ReshapeЛ
$cnn_7/dense_14/MatMul/ReadVariableOpReadVariableOp-cnn_7_dense_14_matmul_readvariableop_resource*
_output_shapes
:	Р@*
dtype02&
$cnn_7/dense_14/MatMul/ReadVariableOpК
cnn_7/dense_14/MatMulMatMul cnn_7/flatten_7/Reshape:output:0,cnn_7/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
cnn_7/dense_14/MatMulЙ
%cnn_7/dense_14/BiasAdd/ReadVariableOpReadVariableOp.cnn_7_dense_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%cnn_7/dense_14/BiasAdd/ReadVariableOpН
cnn_7/dense_14/BiasAddBiasAddcnn_7/dense_14/MatMul:product:0-cnn_7/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
cnn_7/dense_14/BiasAdd
cnn_7/dense_14/ReluRelucnn_7/dense_14/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
cnn_7/dense_14/ReluК
$cnn_7/dense_15/MatMul/ReadVariableOpReadVariableOp-cnn_7_dense_15_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype02&
$cnn_7/dense_15/MatMul/ReadVariableOpЛ
cnn_7/dense_15/MatMulMatMul!cnn_7/dense_14/Relu:activations:0,cnn_7/dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
cnn_7/dense_15/MatMulЙ
%cnn_7/dense_15/BiasAdd/ReadVariableOpReadVariableOp.cnn_7_dense_15_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02'
%cnn_7/dense_15/BiasAdd/ReadVariableOpН
cnn_7/dense_15/BiasAddBiasAddcnn_7/dense_15/MatMul:product:0-cnn_7/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
cnn_7/dense_15/BiasAdd
cnn_7/dense_15/SoftmaxSoftmaxcnn_7/dense_15/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
cnn_7/dense_15/Softmax{
IdentityIdentity cnn_7/dense_15/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identityп
NoOpNoOp'^cnn_7/conv2d_21/BiasAdd/ReadVariableOp&^cnn_7/conv2d_21/Conv2D/ReadVariableOp'^cnn_7/conv2d_22/BiasAdd/ReadVariableOp&^cnn_7/conv2d_22/Conv2D/ReadVariableOp'^cnn_7/conv2d_23/BiasAdd/ReadVariableOp&^cnn_7/conv2d_23/Conv2D/ReadVariableOp&^cnn_7/dense_14/BiasAdd/ReadVariableOp%^cnn_7/dense_14/MatMul/ReadVariableOp&^cnn_7/dense_15/BiasAdd/ReadVariableOp%^cnn_7/dense_15/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ: : : : : : : : : : 2P
&cnn_7/conv2d_21/BiasAdd/ReadVariableOp&cnn_7/conv2d_21/BiasAdd/ReadVariableOp2N
%cnn_7/conv2d_21/Conv2D/ReadVariableOp%cnn_7/conv2d_21/Conv2D/ReadVariableOp2P
&cnn_7/conv2d_22/BiasAdd/ReadVariableOp&cnn_7/conv2d_22/BiasAdd/ReadVariableOp2N
%cnn_7/conv2d_22/Conv2D/ReadVariableOp%cnn_7/conv2d_22/Conv2D/ReadVariableOp2P
&cnn_7/conv2d_23/BiasAdd/ReadVariableOp&cnn_7/conv2d_23/BiasAdd/ReadVariableOp2N
%cnn_7/conv2d_23/Conv2D/ReadVariableOp%cnn_7/conv2d_23/Conv2D/ReadVariableOp2N
%cnn_7/dense_14/BiasAdd/ReadVariableOp%cnn_7/dense_14/BiasAdd/ReadVariableOp2L
$cnn_7/dense_14/MatMul/ReadVariableOp$cnn_7/dense_14/MatMul/ReadVariableOp2N
%cnn_7/dense_15/BiasAdd/ReadVariableOp%cnn_7/dense_15/BiasAdd/ReadVariableOp2L
$cnn_7/dense_15/MatMul/ReadVariableOp$cnn_7/dense_15/MatMul/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
П
h
L__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_205594

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ъ
Ч
D__inference_model_14_layer_call_and_return_conditional_losses_205979	
input&
cnn_7_205957: 
cnn_7_205959: &
cnn_7_205961: @
cnn_7_205963:@&
cnn_7_205965:@@
cnn_7_205967:@
cnn_7_205969:	Р@
cnn_7_205971:@
cnn_7_205973:@

cnn_7_205975:

identityЂcnn_7/StatefulPartitionedCall
cnn_7/StatefulPartitionedCallStatefulPartitionedCallinputcnn_7_205957cnn_7_205959cnn_7_205961cnn_7_205963cnn_7_205965cnn_7_205967cnn_7_205969cnn_7_205971cnn_7_205973cnn_7_205975*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_cnn_7_layer_call_and_return_conditional_losses_2056562
cnn_7/StatefulPartitionedCall
IdentityIdentity&cnn_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identityn
NoOpNoOp^cnn_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ: : : : : : : : : : 2>
cnn_7/StatefulPartitionedCallcnn_7/StatefulPartitionedCall:V R
/
_output_shapes
:џџџџџџџџџ

_user_specified_nameinput
й
M
1__inference_max_pooling2d_14_layer_call_fn_206328

inputs
identityэ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_2055082
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ъ


)__inference_model_14_layer_call_fn_206081

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:	Р@
	unknown_6:@
	unknown_7:@

	unknown_8:

identityЂStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_model_14_layer_call_and_return_conditional_losses_2059062
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ь
ў
E__inference_conv2d_23_layer_call_and_return_conditional_losses_206403

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
б"

__inference__traced_save_206507
file_prefix5
1savev2_cnn_7_conv2d_21_kernel_read_readvariableop3
/savev2_cnn_7_conv2d_21_bias_read_readvariableop5
1savev2_cnn_7_conv2d_22_kernel_read_readvariableop3
/savev2_cnn_7_conv2d_22_bias_read_readvariableop5
1savev2_cnn_7_conv2d_23_kernel_read_readvariableop3
/savev2_cnn_7_conv2d_23_bias_read_readvariableop4
0savev2_cnn_7_dense_14_kernel_read_readvariableop2
.savev2_cnn_7_dense_14_bias_read_readvariableop4
0savev2_cnn_7_dense_15_kernel_read_readvariableop2
.savev2_cnn_7_dense_15_bias_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameБ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*У
valueЙBЖB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
SaveV2/shape_and_slicesД
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_cnn_7_conv2d_21_kernel_read_readvariableop/savev2_cnn_7_conv2d_21_bias_read_readvariableop1savev2_cnn_7_conv2d_22_kernel_read_readvariableop/savev2_cnn_7_conv2d_22_bias_read_readvariableop1savev2_cnn_7_conv2d_23_kernel_read_readvariableop/savev2_cnn_7_conv2d_23_bias_read_readvariableop0savev2_cnn_7_dense_14_kernel_read_readvariableop.savev2_cnn_7_dense_14_bias_read_readvariableop0savev2_cnn_7_dense_15_kernel_read_readvariableop.savev2_cnn_7_dense_15_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*
_input_shapeso
m: : : : @:@:@@:@:	Р@:@:@
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:%!

_output_shapes
:	Р@: 

_output_shapes
:@:$	 

_output_shapes

:@
: 


_output_shapes
:
:

_output_shapes
: 
є9

A__inference_cnn_7_layer_call_and_return_conditional_losses_206303
input_1B
(conv2d_21_conv2d_readvariableop_resource: 7
)conv2d_21_biasadd_readvariableop_resource: B
(conv2d_22_conv2d_readvariableop_resource: @7
)conv2d_22_biasadd_readvariableop_resource:@B
(conv2d_23_conv2d_readvariableop_resource:@@7
)conv2d_23_biasadd_readvariableop_resource:@:
'dense_14_matmul_readvariableop_resource:	Р@6
(dense_14_biasadd_readvariableop_resource:@9
'dense_15_matmul_readvariableop_resource:@
6
(dense_15_biasadd_readvariableop_resource:

identityЂ conv2d_21/BiasAdd/ReadVariableOpЂconv2d_21/Conv2D/ReadVariableOpЂ conv2d_22/BiasAdd/ReadVariableOpЂconv2d_22/Conv2D/ReadVariableOpЂ conv2d_23/BiasAdd/ReadVariableOpЂconv2d_23/Conv2D/ReadVariableOpЂdense_14/BiasAdd/ReadVariableOpЂdense_14/MatMul/ReadVariableOpЂdense_15/BiasAdd/ReadVariableOpЂdense_15/MatMul/ReadVariableOpГ
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_21/Conv2D/ReadVariableOpУ
conv2d_21/Conv2DConv2Dinput_1'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2
conv2d_21/Conv2DЊ
 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_21/BiasAdd/ReadVariableOpА
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
conv2d_21/BiasAdd~
conv2d_21/ReluReluconv2d_21/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
conv2d_21/ReluЪ
max_pooling2d_14/MaxPoolMaxPoolconv2d_21/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_14/MaxPoolГ
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_22/Conv2D/ReadVariableOpн
conv2d_22/Conv2DConv2D!max_pooling2d_14/MaxPool:output:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2
conv2d_22/Conv2DЊ
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_22/BiasAdd/ReadVariableOpА
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv2d_22/BiasAdd~
conv2d_22/ReluReluconv2d_22/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv2d_22/ReluЪ
max_pooling2d_15/MaxPoolMaxPoolconv2d_22/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_15/MaxPoolГ
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_23/Conv2D/ReadVariableOpн
conv2d_23/Conv2DConv2D!max_pooling2d_15/MaxPool:output:0'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2
conv2d_23/Conv2DЊ
 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_23/BiasAdd/ReadVariableOpА
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv2d_23/BiasAdd~
conv2d_23/ReluReluconv2d_23/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv2d_23/Relus
flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@  2
flatten_7/Const
flatten_7/ReshapeReshapeconv2d_23/Relu:activations:0flatten_7/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2
flatten_7/ReshapeЉ
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	Р@*
dtype02 
dense_14/MatMul/ReadVariableOpЂ
dense_14/MatMulMatMulflatten_7/Reshape:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_14/MatMulЇ
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_14/BiasAdd/ReadVariableOpЅ
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_14/BiasAdds
dense_14/ReluReludense_14/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_14/ReluЈ
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype02 
dense_15/MatMul/ReadVariableOpЃ
dense_15/MatMulMatMuldense_14/Relu:activations:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
dense_15/MatMulЇ
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_15/BiasAdd/ReadVariableOpЅ
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
dense_15/BiasAdd|
dense_15/SoftmaxSoftmaxdense_15/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
dense_15/Softmaxu
IdentityIdentitydense_15/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
2

IdentityЃ
NoOpNoOp!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ: : : : : : : : : : 2D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp:X T
/
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
П
h
L__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_205571

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ :W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
П
h
L__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_206383

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
а
F
*__inference_flatten_7_layer_call_fn_206408

inputs
identityФ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_flatten_7_layer_call_and_return_conditional_losses_2056192
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ќ
h
L__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_205508

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ќ
h
L__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_206378

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

ѕ
D__inference_dense_15_layer_call_and_return_conditional_losses_206454

inputs0
matmul_readvariableop_resource:@
-
biasadd_readvariableop_resource:

identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ь
ў
E__inference_conv2d_21_layer_call_and_return_conditional_losses_205561

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ч
a
E__inference_flatten_7_layer_call_and_return_conditional_losses_206414

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ь
ў
E__inference_conv2d_21_layer_call_and_return_conditional_losses_206323

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ь
ў
E__inference_conv2d_23_layer_call_and_return_conditional_losses_205607

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ќ
h
L__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_205530

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


*__inference_conv2d_22_layer_call_fn_206352

inputs!
unknown: @
	unknown_0:@
identityЂStatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_22_layer_call_and_return_conditional_losses_2055842
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
ъ


)__inference_model_14_layer_call_fn_206056

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:	Р@
	unknown_6:@
	unknown_7:@

	unknown_8:

identityЂStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_model_14_layer_call_and_return_conditional_losses_2058312
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


*__inference_conv2d_21_layer_call_fn_206312

inputs!
unknown: 
	unknown_0: 
identityЂStatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_21_layer_call_and_return_conditional_losses_2055612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ь
M
1__inference_max_pooling2d_14_layer_call_fn_206333

inputs
identityв
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_2055712
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ :W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
ь
ў
E__inference_conv2d_22_layer_call_and_return_conditional_losses_206363

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs


*__inference_conv2d_23_layer_call_fn_206392

inputs!
unknown:@@
	unknown_0:@
identityЂStatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_23_layer_call_and_return_conditional_losses_2056072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ч
a
E__inference_flatten_7_layer_call_and_return_conditional_losses_205619

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
т9

A__inference_cnn_7_layer_call_and_return_conditional_losses_206260
xB
(conv2d_21_conv2d_readvariableop_resource: 7
)conv2d_21_biasadd_readvariableop_resource: B
(conv2d_22_conv2d_readvariableop_resource: @7
)conv2d_22_biasadd_readvariableop_resource:@B
(conv2d_23_conv2d_readvariableop_resource:@@7
)conv2d_23_biasadd_readvariableop_resource:@:
'dense_14_matmul_readvariableop_resource:	Р@6
(dense_14_biasadd_readvariableop_resource:@9
'dense_15_matmul_readvariableop_resource:@
6
(dense_15_biasadd_readvariableop_resource:

identityЂ conv2d_21/BiasAdd/ReadVariableOpЂconv2d_21/Conv2D/ReadVariableOpЂ conv2d_22/BiasAdd/ReadVariableOpЂconv2d_22/Conv2D/ReadVariableOpЂ conv2d_23/BiasAdd/ReadVariableOpЂconv2d_23/Conv2D/ReadVariableOpЂdense_14/BiasAdd/ReadVariableOpЂdense_14/MatMul/ReadVariableOpЂdense_15/BiasAdd/ReadVariableOpЂdense_15/MatMul/ReadVariableOpГ
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_21/Conv2D/ReadVariableOpН
conv2d_21/Conv2DConv2Dx'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2
conv2d_21/Conv2DЊ
 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_21/BiasAdd/ReadVariableOpА
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
conv2d_21/BiasAdd~
conv2d_21/ReluReluconv2d_21/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
conv2d_21/ReluЪ
max_pooling2d_14/MaxPoolMaxPoolconv2d_21/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_14/MaxPoolГ
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_22/Conv2D/ReadVariableOpн
conv2d_22/Conv2DConv2D!max_pooling2d_14/MaxPool:output:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2
conv2d_22/Conv2DЊ
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_22/BiasAdd/ReadVariableOpА
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv2d_22/BiasAdd~
conv2d_22/ReluReluconv2d_22/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv2d_22/ReluЪ
max_pooling2d_15/MaxPoolMaxPoolconv2d_22/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_15/MaxPoolГ
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_23/Conv2D/ReadVariableOpн
conv2d_23/Conv2DConv2D!max_pooling2d_15/MaxPool:output:0'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2
conv2d_23/Conv2DЊ
 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_23/BiasAdd/ReadVariableOpА
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv2d_23/BiasAdd~
conv2d_23/ReluReluconv2d_23/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv2d_23/Relus
flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@  2
flatten_7/Const
flatten_7/ReshapeReshapeconv2d_23/Relu:activations:0flatten_7/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2
flatten_7/ReshapeЉ
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	Р@*
dtype02 
dense_14/MatMul/ReadVariableOpЂ
dense_14/MatMulMatMulflatten_7/Reshape:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_14/MatMulЇ
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_14/BiasAdd/ReadVariableOpЅ
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_14/BiasAdds
dense_14/ReluReludense_14/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_14/ReluЈ
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype02 
dense_15/MatMul/ReadVariableOpЃ
dense_15/MatMulMatMuldense_14/Relu:activations:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
dense_15/MatMulЇ
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_15/BiasAdd/ReadVariableOpЅ
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
dense_15/BiasAdd|
dense_15/SoftmaxSoftmaxdense_15/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
dense_15/Softmaxu
IdentityIdentitydense_15/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
2

IdentityЃ
NoOpNoOp!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ: : : : : : : : : : 2D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp:R N
/
_output_shapes
:џџџџџџџџџ

_user_specified_namex

і
D__inference_dense_14_layer_call_and_return_conditional_losses_205632

inputs1
matmul_readvariableop_resource:	Р@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Р@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџР: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs
э
Ш
D__inference_model_14_layer_call_and_return_conditional_losses_205906

inputs&
cnn_7_205884: 
cnn_7_205886: &
cnn_7_205888: @
cnn_7_205890:@&
cnn_7_205892:@@
cnn_7_205894:@
cnn_7_205896:	Р@
cnn_7_205898:@
cnn_7_205900:@

cnn_7_205902:

identityЂcnn_7/StatefulPartitionedCall
cnn_7/StatefulPartitionedCallStatefulPartitionedCallinputscnn_7_205884cnn_7_205886cnn_7_205888cnn_7_205890cnn_7_205892cnn_7_205894cnn_7_205896cnn_7_205898cnn_7_205900cnn_7_205902*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_cnn_7_layer_call_and_return_conditional_losses_2056562
cnn_7/StatefulPartitionedCall
IdentityIdentity&cnn_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identityn
NoOpNoOp^cnn_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ: : : : : : : : : : 2>
cnn_7/StatefulPartitionedCallcnn_7/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ь
ў
E__inference_conv2d_22_layer_call_and_return_conditional_losses_205584

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
'
і
A__inference_cnn_7_layer_call_and_return_conditional_losses_205656
x*
conv2d_21_205562: 
conv2d_21_205564: *
conv2d_22_205585: @
conv2d_22_205587:@*
conv2d_23_205608:@@
conv2d_23_205610:@"
dense_14_205633:	Р@
dense_14_205635:@!
dense_15_205650:@

dense_15_205652:

identityЂ!conv2d_21/StatefulPartitionedCallЂ!conv2d_22/StatefulPartitionedCallЂ!conv2d_23/StatefulPartitionedCallЂ dense_14/StatefulPartitionedCallЂ dense_15/StatefulPartitionedCall
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCallxconv2d_21_205562conv2d_21_205564*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_21_layer_call_and_return_conditional_losses_2055612#
!conv2d_21/StatefulPartitionedCall
 max_pooling2d_14/PartitionedCallPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_2055712"
 max_pooling2d_14/PartitionedCallФ
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_14/PartitionedCall:output:0conv2d_22_205585conv2d_22_205587*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_22_layer_call_and_return_conditional_losses_2055842#
!conv2d_22/StatefulPartitionedCall
 max_pooling2d_15/PartitionedCallPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_2055942"
 max_pooling2d_15/PartitionedCallФ
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_15/PartitionedCall:output:0conv2d_23_205608conv2d_23_205610*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_23_layer_call_and_return_conditional_losses_2056072#
!conv2d_23/StatefulPartitionedCallќ
flatten_7/PartitionedCallPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_flatten_7_layer_call_and_return_conditional_losses_2056192
flatten_7/PartitionedCallА
 dense_14/StatefulPartitionedCallStatefulPartitionedCall"flatten_7/PartitionedCall:output:0dense_14_205633dense_14_205635*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_2056322"
 dense_14/StatefulPartitionedCallЗ
 dense_15/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0dense_15_205650dense_15_205652*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_2056492"
 dense_15/StatefulPartitionedCall
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity
NoOpNoOp"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ: : : : : : : : : : 2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall:R N
/
_output_shapes
:џџџџџџџџџ

_user_specified_namex
П
h
L__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_206343

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ :W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
ч


)__inference_model_14_layer_call_fn_205954	
input!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:	Р@
	unknown_6:@
	unknown_7:@

	unknown_8:

identityЂStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_model_14_layer_call_and_return_conditional_losses_2059062
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:џџџџџџџџџ

_user_specified_nameinput
ы@
	
D__inference_model_14_layer_call_and_return_conditional_losses_206167

inputsH
.cnn_7_conv2d_21_conv2d_readvariableop_resource: =
/cnn_7_conv2d_21_biasadd_readvariableop_resource: H
.cnn_7_conv2d_22_conv2d_readvariableop_resource: @=
/cnn_7_conv2d_22_biasadd_readvariableop_resource:@H
.cnn_7_conv2d_23_conv2d_readvariableop_resource:@@=
/cnn_7_conv2d_23_biasadd_readvariableop_resource:@@
-cnn_7_dense_14_matmul_readvariableop_resource:	Р@<
.cnn_7_dense_14_biasadd_readvariableop_resource:@?
-cnn_7_dense_15_matmul_readvariableop_resource:@
<
.cnn_7_dense_15_biasadd_readvariableop_resource:

identityЂ&cnn_7/conv2d_21/BiasAdd/ReadVariableOpЂ%cnn_7/conv2d_21/Conv2D/ReadVariableOpЂ&cnn_7/conv2d_22/BiasAdd/ReadVariableOpЂ%cnn_7/conv2d_22/Conv2D/ReadVariableOpЂ&cnn_7/conv2d_23/BiasAdd/ReadVariableOpЂ%cnn_7/conv2d_23/Conv2D/ReadVariableOpЂ%cnn_7/dense_14/BiasAdd/ReadVariableOpЂ$cnn_7/dense_14/MatMul/ReadVariableOpЂ%cnn_7/dense_15/BiasAdd/ReadVariableOpЂ$cnn_7/dense_15/MatMul/ReadVariableOpХ
%cnn_7/conv2d_21/Conv2D/ReadVariableOpReadVariableOp.cnn_7_conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02'
%cnn_7/conv2d_21/Conv2D/ReadVariableOpд
cnn_7/conv2d_21/Conv2DConv2Dinputs-cnn_7/conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2
cnn_7/conv2d_21/Conv2DМ
&cnn_7/conv2d_21/BiasAdd/ReadVariableOpReadVariableOp/cnn_7_conv2d_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&cnn_7/conv2d_21/BiasAdd/ReadVariableOpШ
cnn_7/conv2d_21/BiasAddBiasAddcnn_7/conv2d_21/Conv2D:output:0.cnn_7/conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
cnn_7/conv2d_21/BiasAdd
cnn_7/conv2d_21/ReluRelu cnn_7/conv2d_21/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
cnn_7/conv2d_21/Reluм
cnn_7/max_pooling2d_14/MaxPoolMaxPool"cnn_7/conv2d_21/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingVALID*
strides
2 
cnn_7/max_pooling2d_14/MaxPoolХ
%cnn_7/conv2d_22/Conv2D/ReadVariableOpReadVariableOp.cnn_7_conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02'
%cnn_7/conv2d_22/Conv2D/ReadVariableOpѕ
cnn_7/conv2d_22/Conv2DConv2D'cnn_7/max_pooling2d_14/MaxPool:output:0-cnn_7/conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2
cnn_7/conv2d_22/Conv2DМ
&cnn_7/conv2d_22/BiasAdd/ReadVariableOpReadVariableOp/cnn_7_conv2d_22_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&cnn_7/conv2d_22/BiasAdd/ReadVariableOpШ
cnn_7/conv2d_22/BiasAddBiasAddcnn_7/conv2d_22/Conv2D:output:0.cnn_7/conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
cnn_7/conv2d_22/BiasAdd
cnn_7/conv2d_22/ReluRelu cnn_7/conv2d_22/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
cnn_7/conv2d_22/Reluм
cnn_7/max_pooling2d_15/MaxPoolMaxPool"cnn_7/conv2d_22/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides
2 
cnn_7/max_pooling2d_15/MaxPoolХ
%cnn_7/conv2d_23/Conv2D/ReadVariableOpReadVariableOp.cnn_7_conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02'
%cnn_7/conv2d_23/Conv2D/ReadVariableOpѕ
cnn_7/conv2d_23/Conv2DConv2D'cnn_7/max_pooling2d_15/MaxPool:output:0-cnn_7/conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2
cnn_7/conv2d_23/Conv2DМ
&cnn_7/conv2d_23/BiasAdd/ReadVariableOpReadVariableOp/cnn_7_conv2d_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&cnn_7/conv2d_23/BiasAdd/ReadVariableOpШ
cnn_7/conv2d_23/BiasAddBiasAddcnn_7/conv2d_23/Conv2D:output:0.cnn_7/conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
cnn_7/conv2d_23/BiasAdd
cnn_7/conv2d_23/ReluRelu cnn_7/conv2d_23/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
cnn_7/conv2d_23/Relu
cnn_7/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@  2
cnn_7/flatten_7/ConstД
cnn_7/flatten_7/ReshapeReshape"cnn_7/conv2d_23/Relu:activations:0cnn_7/flatten_7/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2
cnn_7/flatten_7/ReshapeЛ
$cnn_7/dense_14/MatMul/ReadVariableOpReadVariableOp-cnn_7_dense_14_matmul_readvariableop_resource*
_output_shapes
:	Р@*
dtype02&
$cnn_7/dense_14/MatMul/ReadVariableOpК
cnn_7/dense_14/MatMulMatMul cnn_7/flatten_7/Reshape:output:0,cnn_7/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
cnn_7/dense_14/MatMulЙ
%cnn_7/dense_14/BiasAdd/ReadVariableOpReadVariableOp.cnn_7_dense_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%cnn_7/dense_14/BiasAdd/ReadVariableOpН
cnn_7/dense_14/BiasAddBiasAddcnn_7/dense_14/MatMul:product:0-cnn_7/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
cnn_7/dense_14/BiasAdd
cnn_7/dense_14/ReluRelucnn_7/dense_14/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
cnn_7/dense_14/ReluК
$cnn_7/dense_15/MatMul/ReadVariableOpReadVariableOp-cnn_7_dense_15_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype02&
$cnn_7/dense_15/MatMul/ReadVariableOpЛ
cnn_7/dense_15/MatMulMatMul!cnn_7/dense_14/Relu:activations:0,cnn_7/dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
cnn_7/dense_15/MatMulЙ
%cnn_7/dense_15/BiasAdd/ReadVariableOpReadVariableOp.cnn_7_dense_15_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02'
%cnn_7/dense_15/BiasAdd/ReadVariableOpН
cnn_7/dense_15/BiasAddBiasAddcnn_7/dense_15/MatMul:product:0-cnn_7/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
cnn_7/dense_15/BiasAdd
cnn_7/dense_15/SoftmaxSoftmaxcnn_7/dense_15/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
cnn_7/dense_15/Softmax{
IdentityIdentity cnn_7/dense_15/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identityп
NoOpNoOp'^cnn_7/conv2d_21/BiasAdd/ReadVariableOp&^cnn_7/conv2d_21/Conv2D/ReadVariableOp'^cnn_7/conv2d_22/BiasAdd/ReadVariableOp&^cnn_7/conv2d_22/Conv2D/ReadVariableOp'^cnn_7/conv2d_23/BiasAdd/ReadVariableOp&^cnn_7/conv2d_23/Conv2D/ReadVariableOp&^cnn_7/dense_14/BiasAdd/ReadVariableOp%^cnn_7/dense_14/MatMul/ReadVariableOp&^cnn_7/dense_15/BiasAdd/ReadVariableOp%^cnn_7/dense_15/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ: : : : : : : : : : 2P
&cnn_7/conv2d_21/BiasAdd/ReadVariableOp&cnn_7/conv2d_21/BiasAdd/ReadVariableOp2N
%cnn_7/conv2d_21/Conv2D/ReadVariableOp%cnn_7/conv2d_21/Conv2D/ReadVariableOp2P
&cnn_7/conv2d_22/BiasAdd/ReadVariableOp&cnn_7/conv2d_22/BiasAdd/ReadVariableOp2N
%cnn_7/conv2d_22/Conv2D/ReadVariableOp%cnn_7/conv2d_22/Conv2D/ReadVariableOp2P
&cnn_7/conv2d_23/BiasAdd/ReadVariableOp&cnn_7/conv2d_23/BiasAdd/ReadVariableOp2N
%cnn_7/conv2d_23/Conv2D/ReadVariableOp%cnn_7/conv2d_23/Conv2D/ReadVariableOp2N
%cnn_7/dense_14/BiasAdd/ReadVariableOp%cnn_7/dense_14/BiasAdd/ReadVariableOp2L
$cnn_7/dense_14/MatMul/ReadVariableOp$cnn_7/dense_14/MatMul/ReadVariableOp2N
%cnn_7/dense_15/BiasAdd/ReadVariableOp%cnn_7/dense_15/BiasAdd/ReadVariableOp2L
$cnn_7/dense_15/MatMul/ReadVariableOp$cnn_7/dense_15/MatMul/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

і
D__inference_dense_14_layer_call_and_return_conditional_losses_206434

inputs1
matmul_readvariableop_resource:	Р@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Р@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџР: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs"ЈL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ќ
serving_default
?
input6
serving_default_input:0џџџџџџџџџ9
cnn_70
StatefulPartitionedCall:0џџџџџџџџџ
tensorflow/serving/predict:В

layer-0
layer_with_weights-0
layer-1
	variables
regularization_losses
trainable_variables
	keras_api

signatures
p__call__
*q&call_and_return_all_conditional_losses
r_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer

	conv1
	maxpooling1
	
conv2
maxpooling2
	conv3
flatten
	dens1
	dens2
	variables
regularization_losses
trainable_variables
	keras_api
s__call__
*t&call_and_return_all_conditional_losses"
_tf_keras_model
f
0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
 "
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
Ъ
layer_regularization_losses
	variables
layer_metrics
regularization_losses
 metrics

!layers
trainable_variables
"non_trainable_variables
p__call__
r_default_save_signature
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
,
userving_default"
signature_map
Л

kernel
bias
#	variables
$regularization_losses
%trainable_variables
&	keras_api
v__call__
*w&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
'	variables
(regularization_losses
)trainable_variables
*	keras_api
x__call__
*y&call_and_return_all_conditional_losses"
_tf_keras_layer
Л

kernel
bias
+	variables
,regularization_losses
-trainable_variables
.	keras_api
z__call__
*{&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
/	variables
0regularization_losses
1trainable_variables
2	keras_api
|__call__
*}&call_and_return_all_conditional_losses"
_tf_keras_layer
Л

kernel
bias
3	variables
4regularization_losses
5trainable_variables
6	keras_api
~__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
Ї
7	variables
8regularization_losses
9trainable_variables
:	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Н

kernel
bias
;	variables
<regularization_losses
=trainable_variables
>	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Н

kernel
bias
?	variables
@regularization_losses
Atrainable_variables
B	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
f
0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
 "
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
­
Clayer_regularization_losses
	variables
Dlayer_metrics
regularization_losses
Emetrics

Flayers
trainable_variables
Gnon_trainable_variables
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
0:. 2cnn_7/conv2d_21/kernel
":  2cnn_7/conv2d_21/bias
0:. @2cnn_7/conv2d_22/kernel
": @2cnn_7/conv2d_22/bias
0:.@@2cnn_7/conv2d_23/kernel
": @2cnn_7/conv2d_23/bias
(:&	Р@2cnn_7/dense_14/kernel
!:@2cnn_7/dense_14/bias
':%@
2cnn_7/dense_15/kernel
!:
2cnn_7/dense_15/bias
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
Hlayer_regularization_losses

Ilayers
#	variables
$regularization_losses
Jmetrics
Klayer_metrics
%trainable_variables
Lnon_trainable_variables
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Mlayer_regularization_losses

Nlayers
'	variables
(regularization_losses
Ometrics
Player_metrics
)trainable_variables
Qnon_trainable_variables
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
Rlayer_regularization_losses

Slayers
+	variables
,regularization_losses
Tmetrics
Ulayer_metrics
-trainable_variables
Vnon_trainable_variables
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Wlayer_regularization_losses

Xlayers
/	variables
0regularization_losses
Ymetrics
Zlayer_metrics
1trainable_variables
[non_trainable_variables
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
\layer_regularization_losses

]layers
3	variables
4regularization_losses
^metrics
_layer_metrics
5trainable_variables
`non_trainable_variables
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
alayer_regularization_losses

blayers
7	variables
8regularization_losses
cmetrics
dlayer_metrics
9trainable_variables
enon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
А
flayer_regularization_losses

glayers
;	variables
<regularization_losses
hmetrics
ilayer_metrics
=trainable_variables
jnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
А
klayer_regularization_losses

llayers
?	variables
@regularization_losses
mmetrics
nlayer_metrics
Atrainable_variables
onon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
X
0
	1

2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
ђ2я
)__inference_model_14_layer_call_fn_205854
)__inference_model_14_layer_call_fn_206056
)__inference_model_14_layer_call_fn_206081
)__inference_model_14_layer_call_fn_205954Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
о2л
D__inference_model_14_layer_call_and_return_conditional_losses_206124
D__inference_model_14_layer_call_and_return_conditional_losses_206167
D__inference_model_14_layer_call_and_return_conditional_losses_205979
D__inference_model_14_layer_call_and_return_conditional_losses_206004Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ЪBЧ
!__inference__wrapped_model_205499input"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓ2№
&__inference_cnn_7_layer_call_fn_206192
&__inference_cnn_7_layer_call_fn_206217
В
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Љ2І
A__inference_cnn_7_layer_call_and_return_conditional_losses_206260
A__inference_cnn_7_layer_call_and_return_conditional_losses_206303
В
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЩBЦ
$__inference_signature_wrapper_206031input"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_conv2d_21_layer_call_fn_206312Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_conv2d_21_layer_call_and_return_conditional_losses_206323Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
1__inference_max_pooling2d_14_layer_call_fn_206328
1__inference_max_pooling2d_14_layer_call_fn_206333Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ф2С
L__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_206338
L__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_206343Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_conv2d_22_layer_call_fn_206352Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_conv2d_22_layer_call_and_return_conditional_losses_206363Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
1__inference_max_pooling2d_15_layer_call_fn_206368
1__inference_max_pooling2d_15_layer_call_fn_206373Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ф2С
L__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_206378
L__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_206383Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_conv2d_23_layer_call_fn_206392Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_conv2d_23_layer_call_and_return_conditional_losses_206403Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_flatten_7_layer_call_fn_206408Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_flatten_7_layer_call_and_return_conditional_losses_206414Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_dense_14_layer_call_fn_206423Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_dense_14_layer_call_and_return_conditional_losses_206434Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_dense_15_layer_call_fn_206443Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_dense_15_layer_call_and_return_conditional_losses_206454Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
!__inference__wrapped_model_205499s
6Ђ3
,Ђ)
'$
inputџџџџџџџџџ
Њ "-Њ*
(
cnn_7
cnn_7џџџџџџџџџ
Ќ
A__inference_cnn_7_layer_call_and_return_conditional_losses_206260g
2Ђ/
(Ђ%
# 
xџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ

 В
A__inference_cnn_7_layer_call_and_return_conditional_losses_206303m
8Ђ5
.Ђ+
)&
input_1џџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ

 
&__inference_cnn_7_layer_call_fn_206192`
8Ђ5
.Ђ+
)&
input_1џџџџџџџџџ
Њ "џџџџџџџџџ

&__inference_cnn_7_layer_call_fn_206217Z
2Ђ/
(Ђ%
# 
xџџџџџџџџџ
Њ "џџџџџџџџџ
Е
E__inference_conv2d_21_layer_call_and_return_conditional_losses_206323l7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "-Ђ*
# 
0џџџџџџџџџ 
 
*__inference_conv2d_21_layer_call_fn_206312_7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ " џџџџџџџџџ Е
E__inference_conv2d_22_layer_call_and_return_conditional_losses_206363l7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ "-Ђ*
# 
0џџџџџџџџџ@
 
*__inference_conv2d_22_layer_call_fn_206352_7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ " џџџџџџџџџ@Е
E__inference_conv2d_23_layer_call_and_return_conditional_losses_206403l7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ "-Ђ*
# 
0џџџџџџџџџ@
 
*__inference_conv2d_23_layer_call_fn_206392_7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ " џџџџџџџџџ@Ѕ
D__inference_dense_14_layer_call_and_return_conditional_losses_206434]0Ђ-
&Ђ#
!
inputsџџџџџџџџџР
Њ "%Ђ"

0џџџџџџџџџ@
 }
)__inference_dense_14_layer_call_fn_206423P0Ђ-
&Ђ#
!
inputsџџџџџџџџџР
Њ "џџџџџџџџџ@Є
D__inference_dense_15_layer_call_and_return_conditional_losses_206454\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "%Ђ"

0џџџџџџџџџ

 |
)__inference_dense_15_layer_call_fn_206443O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "џџџџџџџџџ
Њ
E__inference_flatten_7_layer_call_and_return_conditional_losses_206414a7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ "&Ђ#

0џџџџџџџџџР
 
*__inference_flatten_7_layer_call_fn_206408T7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ "џџџџџџџџџРя
L__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_206338RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 И
L__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_206343h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ "-Ђ*
# 
0џџџџџџџџџ 
 Ч
1__inference_max_pooling2d_14_layer_call_fn_206328RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
1__inference_max_pooling2d_14_layer_call_fn_206333[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ " џџџџџџџџџ я
L__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_206378RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 И
L__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_206383h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ "-Ђ*
# 
0џџџџџџџџџ@
 Ч
1__inference_max_pooling2d_15_layer_call_fn_206368RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
1__inference_max_pooling2d_15_layer_call_fn_206373[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ " џџџџџџџџџ@Л
D__inference_model_14_layer_call_and_return_conditional_losses_205979s
>Ђ;
4Ђ1
'$
inputџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ

 Л
D__inference_model_14_layer_call_and_return_conditional_losses_206004s
>Ђ;
4Ђ1
'$
inputџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ

 М
D__inference_model_14_layer_call_and_return_conditional_losses_206124t
?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ

 М
D__inference_model_14_layer_call_and_return_conditional_losses_206167t
?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ

 
)__inference_model_14_layer_call_fn_205854f
>Ђ;
4Ђ1
'$
inputџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ

)__inference_model_14_layer_call_fn_205954f
>Ђ;
4Ђ1
'$
inputџџџџџџџџџ
p

 
Њ "џџџџџџџџџ

)__inference_model_14_layer_call_fn_206056g
?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ

)__inference_model_14_layer_call_fn_206081g
?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
Є
$__inference_signature_wrapper_206031|
?Ђ<
Ђ 
5Њ2
0
input'$
inputџџџџџџџџџ"-Њ*
(
cnn_7
cnn_7џџџџџџџџџ
