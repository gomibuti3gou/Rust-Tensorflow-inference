??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
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
?
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
?
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
delete_old_dirsbool(?
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
dtypetype?
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
?
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
executor_typestring ?
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02v2.6.0-rc2-32-g919f693420e8??
?
conv2d_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_12/kernel
}
$conv2d_12/kernel/Read/ReadVariableOpReadVariableOpconv2d_12/kernel*&
_output_shapes
: *
dtype0
t
conv2d_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_12/bias
m
"conv2d_12/bias/Read/ReadVariableOpReadVariableOpconv2d_12/bias*
_output_shapes
: *
dtype0
?
conv2d_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_13/kernel
}
$conv2d_13/kernel/Read/ReadVariableOpReadVariableOpconv2d_13/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_13/bias
m
"conv2d_13/bias/Read/ReadVariableOpReadVariableOpconv2d_13/bias*
_output_shapes
: *
dtype0
?
conv2d_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_14/kernel
}
$conv2d_14/kernel/Read/ReadVariableOpReadVariableOpconv2d_14/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_14/bias
m
"conv2d_14/bias/Read/ReadVariableOpReadVariableOpconv2d_14/bias*
_output_shapes
:@*
dtype0
?
conv2d_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_15/kernel
}
$conv2d_15/kernel/Read/ReadVariableOpReadVariableOpconv2d_15/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_15/bias
m
"conv2d_15/bias/Read/ReadVariableOpReadVariableOpconv2d_15/bias*
_output_shapes
:@*
dtype0
z
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
? ?*
shared_namedense_6/kernel
s
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel* 
_output_shapes
:
? ?*
dtype0
q
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_6/bias
j
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes	
:?*
dtype0
y
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*
shared_namedense_7/kernel
r
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes
:	?
*
dtype0
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes
:
*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/conv2d_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_12/kernel/m
?
+Adam/conv2d_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_12/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_12/bias/m
{
)Adam/conv2d_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_12/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_13/kernel/m
?
+Adam/conv2d_13/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_13/kernel/m*&
_output_shapes
:  *
dtype0
?
Adam/conv2d_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_13/bias/m
{
)Adam/conv2d_13/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_13/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_14/kernel/m
?
+Adam/conv2d_14/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_14/kernel/m*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_14/bias/m
{
)Adam/conv2d_14/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_14/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_15/kernel/m
?
+Adam/conv2d_15/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_15/kernel/m*&
_output_shapes
:@@*
dtype0
?
Adam/conv2d_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_15/bias/m
{
)Adam/conv2d_15/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_15/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
? ?*&
shared_nameAdam/dense_6/kernel/m
?
)Adam/dense_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/m* 
_output_shapes
:
? ?*
dtype0

Adam/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_6/bias/m
x
'Adam/dense_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*&
shared_nameAdam/dense_7/kernel/m
?
)Adam/dense_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/m*
_output_shapes
:	?
*
dtype0
~
Adam/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense_7/bias/m
w
'Adam/dense_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/m*
_output_shapes
:
*
dtype0
?
Adam/conv2d_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_12/kernel/v
?
+Adam/conv2d_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_12/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_12/bias/v
{
)Adam/conv2d_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_12/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_13/kernel/v
?
+Adam/conv2d_13/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_13/kernel/v*&
_output_shapes
:  *
dtype0
?
Adam/conv2d_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_13/bias/v
{
)Adam/conv2d_13/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_13/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_14/kernel/v
?
+Adam/conv2d_14/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_14/kernel/v*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_14/bias/v
{
)Adam/conv2d_14/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_14/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_15/kernel/v
?
+Adam/conv2d_15/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_15/kernel/v*&
_output_shapes
:@@*
dtype0
?
Adam/conv2d_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_15/bias/v
{
)Adam/conv2d_15/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_15/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
? ?*&
shared_nameAdam/dense_6/kernel/v
?
)Adam/dense_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/v* 
_output_shapes
:
? ?*
dtype0

Adam/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_6/bias/v
x
'Adam/dense_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*&
shared_nameAdam/dense_7/kernel/v
?
)Adam/dense_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/v*
_output_shapes
:	?
*
dtype0
~
Adam/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense_7/bias/v
w
'Adam/dense_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
?V
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?V
value?UB?U B?U
?
layer-0
layer_with_weights-0
layer-1
trainable_variables
	variables
regularization_losses
	keras_api

signatures
 
?
layer_with_weights-0
layer-0
	layer_with_weights-1
	layer-1

layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
layer-8
layer_with_weights-4
layer-9
layer-10
layer_with_weights-5
layer-11
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api
V
0
1
2
3
4
5
6
 7
!8
"9
#10
$11
V
0
1
2
3
4
5
6
 7
!8
"9
#10
$11
 
?
%layer_metrics
&non_trainable_variables
trainable_variables

'layers
	variables
(layer_regularization_losses
)metrics
regularization_losses
 
h

kernel
bias
*trainable_variables
+	variables
,regularization_losses
-	keras_api
h

kernel
bias
.trainable_variables
/	variables
0regularization_losses
1	keras_api
R
2trainable_variables
3	variables
4regularization_losses
5	keras_api
R
6trainable_variables
7	variables
8regularization_losses
9	keras_api
h

kernel
bias
:trainable_variables
;	variables
<regularization_losses
=	keras_api
h

kernel
 bias
>trainable_variables
?	variables
@regularization_losses
A	keras_api
R
Btrainable_variables
C	variables
Dregularization_losses
E	keras_api
R
Ftrainable_variables
G	variables
Hregularization_losses
I	keras_api
R
Jtrainable_variables
K	variables
Lregularization_losses
M	keras_api
h

!kernel
"bias
Ntrainable_variables
O	variables
Pregularization_losses
Q	keras_api
R
Rtrainable_variables
S	variables
Tregularization_losses
U	keras_api
h

#kernel
$bias
Vtrainable_variables
W	variables
Xregularization_losses
Y	keras_api
?
Ziter

[beta_1

\beta_2
	]decay
^learning_ratem?m?m?m?m?m?m? m?!m?"m?#m?$m?v?v?v?v?v?v?v? v?!v?"v?#v?$v?
V
0
1
2
3
4
5
6
 7
!8
"9
#10
$11
V
0
1
2
3
4
5
6
 7
!8
"9
#10
$11
 
?
_layer_metrics
`non_trainable_variables
trainable_variables

alayers
	variables
blayer_regularization_losses
cmetrics
regularization_losses
VT
VARIABLE_VALUEconv2d_12/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_12/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_13/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_13/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_14/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_14/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_15/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_15/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_6/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_6/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_7/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_7/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1
 
 

0
1

0
1
 
?
dlayer_metrics

elayers
fnon_trainable_variables
*trainable_variables
+	variables
glayer_regularization_losses
,regularization_losses
hmetrics

0
1

0
1
 
?
ilayer_metrics

jlayers
knon_trainable_variables
.trainable_variables
/	variables
llayer_regularization_losses
0regularization_losses
mmetrics
 
 
 
?
nlayer_metrics

olayers
pnon_trainable_variables
2trainable_variables
3	variables
qlayer_regularization_losses
4regularization_losses
rmetrics
 
 
 
?
slayer_metrics

tlayers
unon_trainable_variables
6trainable_variables
7	variables
vlayer_regularization_losses
8regularization_losses
wmetrics

0
1

0
1
 
?
xlayer_metrics

ylayers
znon_trainable_variables
:trainable_variables
;	variables
{layer_regularization_losses
<regularization_losses
|metrics

0
 1

0
 1
 
?
}layer_metrics

~layers
non_trainable_variables
>trainable_variables
?	variables
 ?layer_regularization_losses
@regularization_losses
?metrics
 
 
 
?
?layer_metrics
?layers
?non_trainable_variables
Btrainable_variables
C	variables
 ?layer_regularization_losses
Dregularization_losses
?metrics
 
 
 
?
?layer_metrics
?layers
?non_trainable_variables
Ftrainable_variables
G	variables
 ?layer_regularization_losses
Hregularization_losses
?metrics
 
 
 
?
?layer_metrics
?layers
?non_trainable_variables
Jtrainable_variables
K	variables
 ?layer_regularization_losses
Lregularization_losses
?metrics

!0
"1

!0
"1
 
?
?layer_metrics
?layers
?non_trainable_variables
Ntrainable_variables
O	variables
 ?layer_regularization_losses
Pregularization_losses
?metrics
 
 
 
?
?layer_metrics
?layers
?non_trainable_variables
Rtrainable_variables
S	variables
 ?layer_regularization_losses
Tregularization_losses
?metrics

#0
$1

#0
$1
 
?
?layer_metrics
?layers
?non_trainable_variables
Vtrainable_variables
W	variables
 ?layer_regularization_losses
Xregularization_losses
?metrics
][
VARIABLE_VALUE	Adam/iter>layer_with_weights-0/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEAdam/beta_1@layer_with_weights-0/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEAdam/beta_2@layer_with_weights-0/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE
Adam/decay?layer_with_weights-0/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/learning_rateGlayer_with_weights-0/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
V
0
	1

2
3
4
5
6
7
8
9
10
11
 

?0
?1
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
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
db
VARIABLE_VALUEtotalIlayer_with_weights-0/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEcountIlayer_with_weights-0/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
fd
VARIABLE_VALUEtotal_1Ilayer_with_weights-0/keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEcount_1Ilayer_with_weights-0/keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
??
VARIABLE_VALUEAdam/conv2d_12/kernel/matrainable_variables/0/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_12/bias/matrainable_variables/1/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_13/kernel/matrainable_variables/2/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_13/bias/matrainable_variables/3/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_14/kernel/matrainable_variables/4/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_14/bias/matrainable_variables/5/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_15/kernel/matrainable_variables/6/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_15/bias/matrainable_variables/7/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/dense_6/kernel/matrainable_variables/8/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/dense_6/bias/matrainable_variables/9/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/dense_7/kernel/mbtrainable_variables/10/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/dense_7/bias/mbtrainable_variables/11/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_12/kernel/vatrainable_variables/0/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_12/bias/vatrainable_variables/1/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_13/kernel/vatrainable_variables/2/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_13/bias/vatrainable_variables/3/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_14/kernel/vatrainable_variables/4/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_14/bias/vatrainable_variables/5/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_15/kernel/vatrainable_variables/6/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_15/bias/vatrainable_variables/7/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/dense_6/kernel/vatrainable_variables/8/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/dense_6/bias/vatrainable_variables/9/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/dense_7/kernel/vbtrainable_variables/10/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/dense_7/bias/vbtrainable_variables/11/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_inputPlaceholder*/
_output_shapes
:?????????  *
dtype0*$
shape:?????????  
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputconv2d_12/kernelconv2d_12/biasconv2d_13/kernelconv2d_13/biasconv2d_14/kernelconv2d_14/biasconv2d_15/kernelconv2d_15/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_124249
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_12/kernel/Read/ReadVariableOp"conv2d_12/bias/Read/ReadVariableOp$conv2d_13/kernel/Read/ReadVariableOp"conv2d_13/bias/Read/ReadVariableOp$conv2d_14/kernel/Read/ReadVariableOp"conv2d_14/bias/Read/ReadVariableOp$conv2d_15/kernel/Read/ReadVariableOp"conv2d_15/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv2d_12/kernel/m/Read/ReadVariableOp)Adam/conv2d_12/bias/m/Read/ReadVariableOp+Adam/conv2d_13/kernel/m/Read/ReadVariableOp)Adam/conv2d_13/bias/m/Read/ReadVariableOp+Adam/conv2d_14/kernel/m/Read/ReadVariableOp)Adam/conv2d_14/bias/m/Read/ReadVariableOp+Adam/conv2d_15/kernel/m/Read/ReadVariableOp)Adam/conv2d_15/bias/m/Read/ReadVariableOp)Adam/dense_6/kernel/m/Read/ReadVariableOp'Adam/dense_6/bias/m/Read/ReadVariableOp)Adam/dense_7/kernel/m/Read/ReadVariableOp'Adam/dense_7/bias/m/Read/ReadVariableOp+Adam/conv2d_12/kernel/v/Read/ReadVariableOp)Adam/conv2d_12/bias/v/Read/ReadVariableOp+Adam/conv2d_13/kernel/v/Read/ReadVariableOp)Adam/conv2d_13/bias/v/Read/ReadVariableOp+Adam/conv2d_14/kernel/v/Read/ReadVariableOp)Adam/conv2d_14/bias/v/Read/ReadVariableOp+Adam/conv2d_15/kernel/v/Read/ReadVariableOp)Adam/conv2d_15/bias/v/Read/ReadVariableOp)Adam/dense_6/kernel/v/Read/ReadVariableOp'Adam/dense_6/bias/v/Read/ReadVariableOp)Adam/dense_7/kernel/v/Read/ReadVariableOp'Adam/dense_7/bias/v/Read/ReadVariableOpConst*:
Tin3
12/	*
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
GPU 2J 8? *(
f#R!
__inference__traced_save_125035
?	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_12/kernelconv2d_12/biasconv2d_13/kernelconv2d_13/biasconv2d_14/kernelconv2d_14/biasconv2d_15/kernelconv2d_15/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_12/kernel/mAdam/conv2d_12/bias/mAdam/conv2d_13/kernel/mAdam/conv2d_13/bias/mAdam/conv2d_14/kernel/mAdam/conv2d_14/bias/mAdam/conv2d_15/kernel/mAdam/conv2d_15/bias/mAdam/dense_6/kernel/mAdam/dense_6/bias/mAdam/dense_7/kernel/mAdam/dense_7/bias/mAdam/conv2d_12/kernel/vAdam/conv2d_12/bias/vAdam/conv2d_13/kernel/vAdam/conv2d_13/bias/vAdam/conv2d_14/kernel/vAdam/conv2d_14/bias/vAdam/conv2d_15/kernel/vAdam/conv2d_15/bias/vAdam/dense_6/kernel/vAdam/dense_6/bias/vAdam/dense_7/kernel/vAdam/dense_7/bias/v*9
Tin2
02.*
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
GPU 2J 8? *+
f&R$
"__inference__traced_restore_125180??
?f
?
!__inference__wrapped_model_123410	
inputX
>model_45_sequential_3_conv2d_12_conv2d_readvariableop_resource: M
?model_45_sequential_3_conv2d_12_biasadd_readvariableop_resource: X
>model_45_sequential_3_conv2d_13_conv2d_readvariableop_resource:  M
?model_45_sequential_3_conv2d_13_biasadd_readvariableop_resource: X
>model_45_sequential_3_conv2d_14_conv2d_readvariableop_resource: @M
?model_45_sequential_3_conv2d_14_biasadd_readvariableop_resource:@X
>model_45_sequential_3_conv2d_15_conv2d_readvariableop_resource:@@M
?model_45_sequential_3_conv2d_15_biasadd_readvariableop_resource:@P
<model_45_sequential_3_dense_6_matmul_readvariableop_resource:
? ?L
=model_45_sequential_3_dense_6_biasadd_readvariableop_resource:	?O
<model_45_sequential_3_dense_7_matmul_readvariableop_resource:	?
K
=model_45_sequential_3_dense_7_biasadd_readvariableop_resource:

identity??6model_45/sequential_3/conv2d_12/BiasAdd/ReadVariableOp?5model_45/sequential_3/conv2d_12/Conv2D/ReadVariableOp?6model_45/sequential_3/conv2d_13/BiasAdd/ReadVariableOp?5model_45/sequential_3/conv2d_13/Conv2D/ReadVariableOp?6model_45/sequential_3/conv2d_14/BiasAdd/ReadVariableOp?5model_45/sequential_3/conv2d_14/Conv2D/ReadVariableOp?6model_45/sequential_3/conv2d_15/BiasAdd/ReadVariableOp?5model_45/sequential_3/conv2d_15/Conv2D/ReadVariableOp?4model_45/sequential_3/dense_6/BiasAdd/ReadVariableOp?3model_45/sequential_3/dense_6/MatMul/ReadVariableOp?4model_45/sequential_3/dense_7/BiasAdd/ReadVariableOp?3model_45/sequential_3/dense_7/MatMul/ReadVariableOp?
5model_45/sequential_3/conv2d_12/Conv2D/ReadVariableOpReadVariableOp>model_45_sequential_3_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype027
5model_45/sequential_3/conv2d_12/Conv2D/ReadVariableOp?
&model_45/sequential_3/conv2d_12/Conv2DConv2Dinput=model_45/sequential_3/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2(
&model_45/sequential_3/conv2d_12/Conv2D?
6model_45/sequential_3/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp?model_45_sequential_3_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype028
6model_45/sequential_3/conv2d_12/BiasAdd/ReadVariableOp?
'model_45/sequential_3/conv2d_12/BiasAddBiasAdd/model_45/sequential_3/conv2d_12/Conv2D:output:0>model_45/sequential_3/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2)
'model_45/sequential_3/conv2d_12/BiasAdd?
$model_45/sequential_3/conv2d_12/ReluRelu0model_45/sequential_3/conv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:?????????   2&
$model_45/sequential_3/conv2d_12/Relu?
5model_45/sequential_3/conv2d_13/Conv2D/ReadVariableOpReadVariableOp>model_45_sequential_3_conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype027
5model_45/sequential_3/conv2d_13/Conv2D/ReadVariableOp?
&model_45/sequential_3/conv2d_13/Conv2DConv2D2model_45/sequential_3/conv2d_12/Relu:activations:0=model_45/sequential_3/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2(
&model_45/sequential_3/conv2d_13/Conv2D?
6model_45/sequential_3/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp?model_45_sequential_3_conv2d_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype028
6model_45/sequential_3/conv2d_13/BiasAdd/ReadVariableOp?
'model_45/sequential_3/conv2d_13/BiasAddBiasAdd/model_45/sequential_3/conv2d_13/Conv2D:output:0>model_45/sequential_3/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2)
'model_45/sequential_3/conv2d_13/BiasAdd?
$model_45/sequential_3/conv2d_13/ReluRelu0model_45/sequential_3/conv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:?????????   2&
$model_45/sequential_3/conv2d_13/Relu?
-model_45/sequential_3/max_pooling2d_6/MaxPoolMaxPool2model_45/sequential_3/conv2d_13/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2/
-model_45/sequential_3/max_pooling2d_6/MaxPool?
(model_45/sequential_3/dropout_9/IdentityIdentity6model_45/sequential_3/max_pooling2d_6/MaxPool:output:0*
T0*/
_output_shapes
:????????? 2*
(model_45/sequential_3/dropout_9/Identity?
5model_45/sequential_3/conv2d_14/Conv2D/ReadVariableOpReadVariableOp>model_45_sequential_3_conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype027
5model_45/sequential_3/conv2d_14/Conv2D/ReadVariableOp?
&model_45/sequential_3/conv2d_14/Conv2DConv2D1model_45/sequential_3/dropout_9/Identity:output:0=model_45/sequential_3/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2(
&model_45/sequential_3/conv2d_14/Conv2D?
6model_45/sequential_3/conv2d_14/BiasAdd/ReadVariableOpReadVariableOp?model_45_sequential_3_conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6model_45/sequential_3/conv2d_14/BiasAdd/ReadVariableOp?
'model_45/sequential_3/conv2d_14/BiasAddBiasAdd/model_45/sequential_3/conv2d_14/Conv2D:output:0>model_45/sequential_3/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2)
'model_45/sequential_3/conv2d_14/BiasAdd?
$model_45/sequential_3/conv2d_14/ReluRelu0model_45/sequential_3/conv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2&
$model_45/sequential_3/conv2d_14/Relu?
5model_45/sequential_3/conv2d_15/Conv2D/ReadVariableOpReadVariableOp>model_45_sequential_3_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype027
5model_45/sequential_3/conv2d_15/Conv2D/ReadVariableOp?
&model_45/sequential_3/conv2d_15/Conv2DConv2D2model_45/sequential_3/conv2d_14/Relu:activations:0=model_45/sequential_3/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2(
&model_45/sequential_3/conv2d_15/Conv2D?
6model_45/sequential_3/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp?model_45_sequential_3_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6model_45/sequential_3/conv2d_15/BiasAdd/ReadVariableOp?
'model_45/sequential_3/conv2d_15/BiasAddBiasAdd/model_45/sequential_3/conv2d_15/Conv2D:output:0>model_45/sequential_3/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2)
'model_45/sequential_3/conv2d_15/BiasAdd?
$model_45/sequential_3/conv2d_15/ReluRelu0model_45/sequential_3/conv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2&
$model_45/sequential_3/conv2d_15/Relu?
-model_45/sequential_3/max_pooling2d_7/MaxPoolMaxPool2model_45/sequential_3/conv2d_15/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2/
-model_45/sequential_3/max_pooling2d_7/MaxPool?
)model_45/sequential_3/dropout_10/IdentityIdentity6model_45/sequential_3/max_pooling2d_7/MaxPool:output:0*
T0*/
_output_shapes
:?????????@2+
)model_45/sequential_3/dropout_10/Identity?
%model_45/sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2'
%model_45/sequential_3/flatten_3/Const?
'model_45/sequential_3/flatten_3/ReshapeReshape2model_45/sequential_3/dropout_10/Identity:output:0.model_45/sequential_3/flatten_3/Const:output:0*
T0*(
_output_shapes
:?????????? 2)
'model_45/sequential_3/flatten_3/Reshape?
3model_45/sequential_3/dense_6/MatMul/ReadVariableOpReadVariableOp<model_45_sequential_3_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype025
3model_45/sequential_3/dense_6/MatMul/ReadVariableOp?
$model_45/sequential_3/dense_6/MatMulMatMul0model_45/sequential_3/flatten_3/Reshape:output:0;model_45/sequential_3/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2&
$model_45/sequential_3/dense_6/MatMul?
4model_45/sequential_3/dense_6/BiasAdd/ReadVariableOpReadVariableOp=model_45_sequential_3_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype026
4model_45/sequential_3/dense_6/BiasAdd/ReadVariableOp?
%model_45/sequential_3/dense_6/BiasAddBiasAdd.model_45/sequential_3/dense_6/MatMul:product:0<model_45/sequential_3/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2'
%model_45/sequential_3/dense_6/BiasAdd?
"model_45/sequential_3/dense_6/ReluRelu.model_45/sequential_3/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2$
"model_45/sequential_3/dense_6/Relu?
)model_45/sequential_3/dropout_11/IdentityIdentity0model_45/sequential_3/dense_6/Relu:activations:0*
T0*(
_output_shapes
:??????????2+
)model_45/sequential_3/dropout_11/Identity?
3model_45/sequential_3/dense_7/MatMul/ReadVariableOpReadVariableOp<model_45_sequential_3_dense_7_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype025
3model_45/sequential_3/dense_7/MatMul/ReadVariableOp?
$model_45/sequential_3/dense_7/MatMulMatMul2model_45/sequential_3/dropout_11/Identity:output:0;model_45/sequential_3/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2&
$model_45/sequential_3/dense_7/MatMul?
4model_45/sequential_3/dense_7/BiasAdd/ReadVariableOpReadVariableOp=model_45_sequential_3_dense_7_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype026
4model_45/sequential_3/dense_7/BiasAdd/ReadVariableOp?
%model_45/sequential_3/dense_7/BiasAddBiasAdd.model_45/sequential_3/dense_7/MatMul:product:0<model_45/sequential_3/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2'
%model_45/sequential_3/dense_7/BiasAdd?
%model_45/sequential_3/dense_7/SoftmaxSoftmax.model_45/sequential_3/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2'
%model_45/sequential_3/dense_7/Softmax?
IdentityIdentity/model_45/sequential_3/dense_7/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp7^model_45/sequential_3/conv2d_12/BiasAdd/ReadVariableOp6^model_45/sequential_3/conv2d_12/Conv2D/ReadVariableOp7^model_45/sequential_3/conv2d_13/BiasAdd/ReadVariableOp6^model_45/sequential_3/conv2d_13/Conv2D/ReadVariableOp7^model_45/sequential_3/conv2d_14/BiasAdd/ReadVariableOp6^model_45/sequential_3/conv2d_14/Conv2D/ReadVariableOp7^model_45/sequential_3/conv2d_15/BiasAdd/ReadVariableOp6^model_45/sequential_3/conv2d_15/Conv2D/ReadVariableOp5^model_45/sequential_3/dense_6/BiasAdd/ReadVariableOp4^model_45/sequential_3/dense_6/MatMul/ReadVariableOp5^model_45/sequential_3/dense_7/BiasAdd/ReadVariableOp4^model_45/sequential_3/dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????  : : : : : : : : : : : : 2p
6model_45/sequential_3/conv2d_12/BiasAdd/ReadVariableOp6model_45/sequential_3/conv2d_12/BiasAdd/ReadVariableOp2n
5model_45/sequential_3/conv2d_12/Conv2D/ReadVariableOp5model_45/sequential_3/conv2d_12/Conv2D/ReadVariableOp2p
6model_45/sequential_3/conv2d_13/BiasAdd/ReadVariableOp6model_45/sequential_3/conv2d_13/BiasAdd/ReadVariableOp2n
5model_45/sequential_3/conv2d_13/Conv2D/ReadVariableOp5model_45/sequential_3/conv2d_13/Conv2D/ReadVariableOp2p
6model_45/sequential_3/conv2d_14/BiasAdd/ReadVariableOp6model_45/sequential_3/conv2d_14/BiasAdd/ReadVariableOp2n
5model_45/sequential_3/conv2d_14/Conv2D/ReadVariableOp5model_45/sequential_3/conv2d_14/Conv2D/ReadVariableOp2p
6model_45/sequential_3/conv2d_15/BiasAdd/ReadVariableOp6model_45/sequential_3/conv2d_15/BiasAdd/ReadVariableOp2n
5model_45/sequential_3/conv2d_15/Conv2D/ReadVariableOp5model_45/sequential_3/conv2d_15/Conv2D/ReadVariableOp2l
4model_45/sequential_3/dense_6/BiasAdd/ReadVariableOp4model_45/sequential_3/dense_6/BiasAdd/ReadVariableOp2j
3model_45/sequential_3/dense_6/MatMul/ReadVariableOp3model_45/sequential_3/dense_6/MatMul/ReadVariableOp2l
4model_45/sequential_3/dense_7/BiasAdd/ReadVariableOp4model_45/sequential_3/dense_7/BiasAdd/ReadVariableOp2j
3model_45/sequential_3/dense_7/MatMul/ReadVariableOp3model_45/sequential_3/dense_7/MatMul/ReadVariableOp:V R
/
_output_shapes
:?????????  

_user_specified_nameinput
?
L
0__inference_max_pooling2d_7_layer_call_fn_124767

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_1234412
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
D__inference_model_45_layer_call_and_return_conditional_losses_124189	
input-
sequential_3_124163: !
sequential_3_124165: -
sequential_3_124167:  !
sequential_3_124169: -
sequential_3_124171: @!
sequential_3_124173:@-
sequential_3_124175:@@!
sequential_3_124177:@'
sequential_3_124179:
? ?"
sequential_3_124181:	?&
sequential_3_124183:	?
!
sequential_3_124185:

identity??$sequential_3/StatefulPartitionedCall?
$sequential_3/StatefulPartitionedCallStatefulPartitionedCallinputsequential_3_124163sequential_3_124165sequential_3_124167sequential_3_124169sequential_3_124171sequential_3_124173sequential_3_124175sequential_3_124177sequential_3_124179sequential_3_124181sequential_3_124183sequential_3_124185*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_1236052&
$sequential_3/StatefulPartitionedCall?
IdentityIdentity-sequential_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityu
NoOpNoOp%^sequential_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????  : : : : : : : : : : : : 2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall:V R
/
_output_shapes
:?????????  

_user_specified_nameinput
?
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_123561

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????? 2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
e
F__inference_dropout_11_layer_call_and_return_conditional_losses_124847

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_123546

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
E__inference_conv2d_15_layer_call_and_return_conditional_losses_123536

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
*__inference_conv2d_15_layer_call_fn_124752

inputs!
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_1235362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_124249	
input!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:
? ?
	unknown_8:	?
	unknown_9:	?


unknown_10:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_1234102
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????  : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:?????????  

_user_specified_nameinput
?
?
-__inference_sequential_3_layer_call_fn_123632
conv2d_12_input!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:
? ?
	unknown_8:	?
	unknown_9:	?


unknown_10:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_1236052
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????  : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????  
)
_user_specified_nameconv2d_12_input
?
d
F__inference_dropout_11_layer_call_and_return_conditional_losses_124835

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_14_layer_call_and_return_conditional_losses_124723

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
d
F__inference_dropout_10_layer_call_and_return_conditional_losses_123553

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
E__inference_conv2d_13_layer_call_and_return_conditional_losses_124656

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????   2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????   2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
)__inference_model_45_layer_call_fn_124405

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:
? ?
	unknown_8:	?
	unknown_9:	?


unknown_10:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_45_layer_call_and_return_conditional_losses_1240172
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????  : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
L
0__inference_max_pooling2d_6_layer_call_fn_124680

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_1234192
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_9_layer_call_and_return_conditional_losses_123506

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:????????? 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:????????? 2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
D__inference_model_45_layer_call_and_return_conditional_losses_124218	
input-
sequential_3_124192: !
sequential_3_124194: -
sequential_3_124196:  !
sequential_3_124198: -
sequential_3_124200: @!
sequential_3_124202:@-
sequential_3_124204:@@!
sequential_3_124206:@'
sequential_3_124208:
? ?"
sequential_3_124210:	?&
sequential_3_124212:	?
!
sequential_3_124214:

identity??$sequential_3/StatefulPartitionedCall?
$sequential_3/StatefulPartitionedCallStatefulPartitionedCallinputsequential_3_124192sequential_3_124194sequential_3_124196sequential_3_124198sequential_3_124200sequential_3_124202sequential_3_124204sequential_3_124206sequential_3_124208sequential_3_124210sequential_3_124212sequential_3_124214*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_1238482&
$sequential_3/StatefulPartitionedCall?
IdentityIdentity-sequential_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityu
NoOpNoOp%^sequential_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????  : : : : : : : : : : : : 2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall:V R
/
_output_shapes
:?????????  

_user_specified_nameinput
?
G
+__inference_dropout_10_layer_call_fn_124794

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_1235532
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
D__inference_model_45_layer_call_and_return_conditional_losses_124104

inputs-
sequential_3_124078: !
sequential_3_124080: -
sequential_3_124082:  !
sequential_3_124084: -
sequential_3_124086: @!
sequential_3_124088:@-
sequential_3_124090:@@!
sequential_3_124092:@'
sequential_3_124094:
? ?"
sequential_3_124096:	?&
sequential_3_124098:	?
!
sequential_3_124100:

identity??$sequential_3/StatefulPartitionedCall?
$sequential_3/StatefulPartitionedCallStatefulPartitionedCallinputssequential_3_124078sequential_3_124080sequential_3_124082sequential_3_124084sequential_3_124086sequential_3_124088sequential_3_124090sequential_3_124092sequential_3_124094sequential_3_124096sequential_3_124098sequential_3_124100*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_1238482&
$sequential_3/StatefulPartitionedCall?
IdentityIdentity-sequential_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityu
NoOpNoOp%^sequential_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????  : : : : : : : : : : : : 2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
d
E__inference_dropout_9_layer_call_and_return_conditional_losses_124702

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:????????? 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:????????? *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? 2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? 2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:????????? 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
e
F__inference_dropout_10_layer_call_and_return_conditional_losses_124789

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
d
E__inference_dropout_9_layer_call_and_return_conditional_losses_123749

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:????????? 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:????????? *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? 2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? 2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:????????? 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
F
*__inference_dropout_9_layer_call_fn_124707

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_1235062
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?F
?	
H__inference_sequential_3_layer_call_and_return_conditional_losses_124493

inputsB
(conv2d_12_conv2d_readvariableop_resource: 7
)conv2d_12_biasadd_readvariableop_resource: B
(conv2d_13_conv2d_readvariableop_resource:  7
)conv2d_13_biasadd_readvariableop_resource: B
(conv2d_14_conv2d_readvariableop_resource: @7
)conv2d_14_biasadd_readvariableop_resource:@B
(conv2d_15_conv2d_readvariableop_resource:@@7
)conv2d_15_biasadd_readvariableop_resource:@:
&dense_6_matmul_readvariableop_resource:
? ?6
'dense_6_biasadd_readvariableop_resource:	?9
&dense_7_matmul_readvariableop_resource:	?
5
'dense_7_biasadd_readvariableop_resource:

identity?? conv2d_12/BiasAdd/ReadVariableOp?conv2d_12/Conv2D/ReadVariableOp? conv2d_13/BiasAdd/ReadVariableOp?conv2d_13/Conv2D/ReadVariableOp? conv2d_14/BiasAdd/ReadVariableOp?conv2d_14/Conv2D/ReadVariableOp? conv2d_15/BiasAdd/ReadVariableOp?conv2d_15/Conv2D/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?dense_7/BiasAdd/ReadVariableOp?dense_7/MatMul/ReadVariableOp?
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_12/Conv2D/ReadVariableOp?
conv2d_12/Conv2DConv2Dinputs'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
conv2d_12/Conv2D?
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_12/BiasAdd/ReadVariableOp?
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
conv2d_12/BiasAdd~
conv2d_12/ReluReluconv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:?????????   2
conv2d_12/Relu?
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_13/Conv2D/ReadVariableOp?
conv2d_13/Conv2DConv2Dconv2d_12/Relu:activations:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
conv2d_13/Conv2D?
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_13/BiasAdd/ReadVariableOp?
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
conv2d_13/BiasAdd~
conv2d_13/ReluReluconv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:?????????   2
conv2d_13/Relu?
max_pooling2d_6/MaxPoolMaxPoolconv2d_13/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_6/MaxPool?
dropout_9/IdentityIdentity max_pooling2d_6/MaxPool:output:0*
T0*/
_output_shapes
:????????? 2
dropout_9/Identity?
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_14/Conv2D/ReadVariableOp?
conv2d_14/Conv2DConv2Ddropout_9/Identity:output:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_14/Conv2D?
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_14/BiasAdd/ReadVariableOp?
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_14/BiasAdd~
conv2d_14/ReluReluconv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_14/Relu?
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_15/Conv2D/ReadVariableOp?
conv2d_15/Conv2DConv2Dconv2d_14/Relu:activations:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_15/Conv2D?
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_15/BiasAdd/ReadVariableOp?
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_15/BiasAdd~
conv2d_15/ReluReluconv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_15/Relu?
max_pooling2d_7/MaxPoolMaxPoolconv2d_15/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_7/MaxPool?
dropout_10/IdentityIdentity max_pooling2d_7/MaxPool:output:0*
T0*/
_output_shapes
:?????????@2
dropout_10/Identitys
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_3/Const?
flatten_3/ReshapeReshapedropout_10/Identity:output:0flatten_3/Const:output:0*
T0*(
_output_shapes
:?????????? 2
flatten_3/Reshape?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMulflatten_3/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_6/BiasAddq
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_6/Relu?
dropout_11/IdentityIdentitydense_6/Relu:activations:0*
T0*(
_output_shapes
:??????????2
dropout_11/Identity?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMuldropout_11/Identity:output:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_7/MatMul?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_7/BiasAddy
dense_7/SoftmaxSoftmaxdense_7/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_7/Softmaxt
IdentityIdentitydense_7/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????  : : : : : : : : : : : : 2D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
E__inference_conv2d_12_layer_call_and_return_conditional_losses_123472

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????   2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????   2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_124805

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????? 2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?:
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_123984
conv2d_12_input*
conv2d_12_123947: 
conv2d_12_123949: *
conv2d_13_123952:  
conv2d_13_123954: *
conv2d_14_123959: @
conv2d_14_123961:@*
conv2d_15_123964:@@
conv2d_15_123966:@"
dense_6_123972:
? ?
dense_6_123974:	?!
dense_7_123978:	?

dense_7_123980:

identity??!conv2d_12/StatefulPartitionedCall?!conv2d_13/StatefulPartitionedCall?!conv2d_14/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?"dropout_10/StatefulPartitionedCall?"dropout_11/StatefulPartitionedCall?!dropout_9/StatefulPartitionedCall?
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCallconv2d_12_inputconv2d_12_123947conv2d_12_123949*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_1234722#
!conv2d_12/StatefulPartitionedCall?
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_123952conv2d_13_123954*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_1234892#
!conv2d_13/StatefulPartitionedCall?
max_pooling2d_6/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_1234992!
max_pooling2d_6/PartitionedCall?
!dropout_9/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_1237492#
!dropout_9/StatefulPartitionedCall?
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall*dropout_9/StatefulPartitionedCall:output:0conv2d_14_123959conv2d_14_123961*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_1235192#
!conv2d_14/StatefulPartitionedCall?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0conv2d_15_123964conv2d_15_123966*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_1235362#
!conv2d_15/StatefulPartitionedCall?
max_pooling2d_7/PartitionedCallPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_1235462!
max_pooling2d_7/PartitionedCall?
"dropout_10/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_7/PartitionedCall:output:0"^dropout_9/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_1237012$
"dropout_10/StatefulPartitionedCall?
flatten_3/PartitionedCallPartitionedCall+dropout_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_1235612
flatten_3/PartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_6_123972dense_6_123974*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_1235742!
dense_6/StatefulPartitionedCall?
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0#^dropout_10/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_1236622$
"dropout_11/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall+dropout_11/StatefulPartitionedCall:output:0dense_7_123978dense_7_123980*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_1235982!
dense_7/StatefulPartitionedCall?
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall#^dropout_10/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????  : : : : : : : : : : : : 2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall:` \
/
_output_shapes
:?????????  
)
_user_specified_nameconv2d_12_input
?
?
-__inference_sequential_3_layer_call_fn_124596

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:
? ?
	unknown_8:	?
	unknown_9:	?


unknown_10:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_1236052
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????  : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
e
F__inference_dropout_11_layer_call_and_return_conditional_losses_123662

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
)__inference_model_45_layer_call_fn_124434

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:
? ?
	unknown_8:	?
	unknown_9:	?


unknown_10:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_45_layer_call_and_return_conditional_losses_1241042
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????  : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_124670

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?:
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_123848

inputs*
conv2d_12_123811: 
conv2d_12_123813: *
conv2d_13_123816:  
conv2d_13_123818: *
conv2d_14_123823: @
conv2d_14_123825:@*
conv2d_15_123828:@@
conv2d_15_123830:@"
dense_6_123836:
? ?
dense_6_123838:	?!
dense_7_123842:	?

dense_7_123844:

identity??!conv2d_12/StatefulPartitionedCall?!conv2d_13/StatefulPartitionedCall?!conv2d_14/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?"dropout_10/StatefulPartitionedCall?"dropout_11/StatefulPartitionedCall?!dropout_9/StatefulPartitionedCall?
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_12_123811conv2d_12_123813*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_1234722#
!conv2d_12/StatefulPartitionedCall?
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_123816conv2d_13_123818*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_1234892#
!conv2d_13/StatefulPartitionedCall?
max_pooling2d_6/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_1234992!
max_pooling2d_6/PartitionedCall?
!dropout_9/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_1237492#
!dropout_9/StatefulPartitionedCall?
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall*dropout_9/StatefulPartitionedCall:output:0conv2d_14_123823conv2d_14_123825*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_1235192#
!conv2d_14/StatefulPartitionedCall?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0conv2d_15_123828conv2d_15_123830*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_1235362#
!conv2d_15/StatefulPartitionedCall?
max_pooling2d_7/PartitionedCallPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_1235462!
max_pooling2d_7/PartitionedCall?
"dropout_10/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_7/PartitionedCall:output:0"^dropout_9/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_1237012$
"dropout_10/StatefulPartitionedCall?
flatten_3/PartitionedCallPartitionedCall+dropout_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_1235612
flatten_3/PartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_6_123836dense_6_123838*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_1235742!
dense_6/StatefulPartitionedCall?
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0#^dropout_10/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_1236622$
"dropout_11/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall+dropout_11/StatefulPartitionedCall:output:0dense_7_123842dense_7_123844*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_1235982!
dense_7/StatefulPartitionedCall?
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall#^dropout_10/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????  : : : : : : : : : : : : 2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
D__inference_model_45_layer_call_and_return_conditional_losses_124017

inputs-
sequential_3_123991: !
sequential_3_123993: -
sequential_3_123995:  !
sequential_3_123997: -
sequential_3_123999: @!
sequential_3_124001:@-
sequential_3_124003:@@!
sequential_3_124005:@'
sequential_3_124007:
? ?"
sequential_3_124009:	?&
sequential_3_124011:	?
!
sequential_3_124013:

identity??$sequential_3/StatefulPartitionedCall?
$sequential_3/StatefulPartitionedCallStatefulPartitionedCallinputssequential_3_123991sequential_3_123993sequential_3_123995sequential_3_123997sequential_3_123999sequential_3_124001sequential_3_124003sequential_3_124005sequential_3_124007sequential_3_124009sequential_3_124011sequential_3_124013*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_1236052&
$sequential_3/StatefulPartitionedCall?
IdentityIdentity-sequential_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityu
NoOpNoOp%^sequential_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????  : : : : : : : : : : : : 2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
*__inference_conv2d_14_layer_call_fn_124732

inputs!
unknown: @
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_1235192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?5
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_123605

inputs*
conv2d_12_123473: 
conv2d_12_123475: *
conv2d_13_123490:  
conv2d_13_123492: *
conv2d_14_123520: @
conv2d_14_123522:@*
conv2d_15_123537:@@
conv2d_15_123539:@"
dense_6_123575:
? ?
dense_6_123577:	?!
dense_7_123599:	?

dense_7_123601:

identity??!conv2d_12/StatefulPartitionedCall?!conv2d_13/StatefulPartitionedCall?!conv2d_14/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_12_123473conv2d_12_123475*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_1234722#
!conv2d_12/StatefulPartitionedCall?
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_123490conv2d_13_123492*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_1234892#
!conv2d_13/StatefulPartitionedCall?
max_pooling2d_6/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_1234992!
max_pooling2d_6/PartitionedCall?
dropout_9/PartitionedCallPartitionedCall(max_pooling2d_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_1235062
dropout_9/PartitionedCall?
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall"dropout_9/PartitionedCall:output:0conv2d_14_123520conv2d_14_123522*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_1235192#
!conv2d_14/StatefulPartitionedCall?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0conv2d_15_123537conv2d_15_123539*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_1235362#
!conv2d_15/StatefulPartitionedCall?
max_pooling2d_7/PartitionedCallPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_1235462!
max_pooling2d_7/PartitionedCall?
dropout_10/PartitionedCallPartitionedCall(max_pooling2d_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_1235532
dropout_10/PartitionedCall?
flatten_3/PartitionedCallPartitionedCall#dropout_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_1235612
flatten_3/PartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_6_123575dense_6_123577*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_1235742!
dense_6/StatefulPartitionedCall?
dropout_11/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_1235852
dropout_11/PartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0dense_7_123599dense_7_123601*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_1235982!
dense_7/StatefulPartitionedCall?
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????  : : : : : : : : : : : : 2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
*__inference_conv2d_12_layer_call_fn_124645

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_1234722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????   2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_124757

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_14_layer_call_and_return_conditional_losses_123519

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
E__inference_conv2d_15_layer_call_and_return_conditional_losses_124743

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_123499

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????   :W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
)__inference_model_45_layer_call_fn_124044	
input!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:
? ?
	unknown_8:	?
	unknown_9:	?


unknown_10:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_45_layer_call_and_return_conditional_losses_1240172
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????  : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:?????????  

_user_specified_nameinput
?
?
E__inference_conv2d_12_layer_call_and_return_conditional_losses_124636

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????   2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????   2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
F
*__inference_flatten_3_layer_call_fn_124810

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_1235612
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
C__inference_dense_7_layer_call_and_return_conditional_losses_124868

inputs1
matmul_readvariableop_resource:	?
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
"__inference__traced_restore_125180
file_prefix;
!assignvariableop_conv2d_12_kernel: /
!assignvariableop_1_conv2d_12_bias: =
#assignvariableop_2_conv2d_13_kernel:  /
!assignvariableop_3_conv2d_13_bias: =
#assignvariableop_4_conv2d_14_kernel: @/
!assignvariableop_5_conv2d_14_bias:@=
#assignvariableop_6_conv2d_15_kernel:@@/
!assignvariableop_7_conv2d_15_bias:@5
!assignvariableop_8_dense_6_kernel:
? ?.
assignvariableop_9_dense_6_bias:	?5
"assignvariableop_10_dense_7_kernel:	?
.
 assignvariableop_11_dense_7_bias:
'
assignvariableop_12_adam_iter:	 )
assignvariableop_13_adam_beta_1: )
assignvariableop_14_adam_beta_2: (
assignvariableop_15_adam_decay: 0
&assignvariableop_16_adam_learning_rate: #
assignvariableop_17_total: #
assignvariableop_18_count: %
assignvariableop_19_total_1: %
assignvariableop_20_count_1: E
+assignvariableop_21_adam_conv2d_12_kernel_m: 7
)assignvariableop_22_adam_conv2d_12_bias_m: E
+assignvariableop_23_adam_conv2d_13_kernel_m:  7
)assignvariableop_24_adam_conv2d_13_bias_m: E
+assignvariableop_25_adam_conv2d_14_kernel_m: @7
)assignvariableop_26_adam_conv2d_14_bias_m:@E
+assignvariableop_27_adam_conv2d_15_kernel_m:@@7
)assignvariableop_28_adam_conv2d_15_bias_m:@=
)assignvariableop_29_adam_dense_6_kernel_m:
? ?6
'assignvariableop_30_adam_dense_6_bias_m:	?<
)assignvariableop_31_adam_dense_7_kernel_m:	?
5
'assignvariableop_32_adam_dense_7_bias_m:
E
+assignvariableop_33_adam_conv2d_12_kernel_v: 7
)assignvariableop_34_adam_conv2d_12_bias_v: E
+assignvariableop_35_adam_conv2d_13_kernel_v:  7
)assignvariableop_36_adam_conv2d_13_bias_v: E
+assignvariableop_37_adam_conv2d_14_kernel_v: @7
)assignvariableop_38_adam_conv2d_14_bias_v:@E
+assignvariableop_39_adam_conv2d_15_kernel_v:@@7
)assignvariableop_40_adam_conv2d_15_bias_v:@=
)assignvariableop_41_adam_dense_6_kernel_v:
? ?6
'assignvariableop_42_adam_dense_6_bias_v:	?<
)assignvariableop_43_adam_dense_7_kernel_v:	?
5
'assignvariableop_44_adam_dense_7_bias_v:

identity_46??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*?
value?B?.B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-0/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-0/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-0/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-0/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-0/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-0/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-0/keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-0/keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/0/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/1/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/2/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/3/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/4/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/5/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/6/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/7/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/8/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/9/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBbtrainable_variables/10/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBbtrainable_variables/11/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/0/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/1/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/2/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/3/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/4/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/5/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/6/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/7/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/8/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/9/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBbtrainable_variables/10/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBbtrainable_variables/11/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_12_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_12_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_13_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_13_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_14_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_14_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_15_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_15_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_6_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_6_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_7_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_7_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_conv2d_12_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_conv2d_12_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_conv2d_13_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_conv2d_13_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_conv2d_14_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_conv2d_14_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_conv2d_15_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_conv2d_15_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_6_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_6_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_7_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_7_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_12_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_12_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_conv2d_13_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_conv2d_13_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_conv2d_14_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_conv2d_14_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_conv2d_15_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_conv2d_15_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp)assignvariableop_41_adam_dense_6_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp'assignvariableop_42_adam_dense_6_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_dense_7_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp'assignvariableop_44_adam_dense_7_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_449
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_45Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_45f
Identity_46IdentityIdentity_45:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_46?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_46Identity_46:output:0*o
_input_shapes^
\: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442(
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
?
d
+__inference_dropout_10_layer_call_fn_124799

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_1237012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
)__inference_model_45_layer_call_fn_124160	
input!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:
? ?
	unknown_8:	?
	unknown_9:	?


unknown_10:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_45_layer_call_and_return_conditional_losses_1241042
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????  : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:?????????  

_user_specified_nameinput
?
d
+__inference_dropout_11_layer_call_fn_124857

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_1236622
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
L
0__inference_max_pooling2d_6_layer_call_fn_124685

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_1234992
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????   :W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
E__inference_conv2d_13_layer_call_and_return_conditional_losses_123489

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????   2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????   2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
c
*__inference_dropout_9_layer_call_fn_124712

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_1237492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
C__inference_dense_6_layer_call_and_return_conditional_losses_123574

inputs2
matmul_readvariableop_resource:
? ?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?Y
?
D__inference_model_45_layer_call_and_return_conditional_losses_124302

inputsO
5sequential_3_conv2d_12_conv2d_readvariableop_resource: D
6sequential_3_conv2d_12_biasadd_readvariableop_resource: O
5sequential_3_conv2d_13_conv2d_readvariableop_resource:  D
6sequential_3_conv2d_13_biasadd_readvariableop_resource: O
5sequential_3_conv2d_14_conv2d_readvariableop_resource: @D
6sequential_3_conv2d_14_biasadd_readvariableop_resource:@O
5sequential_3_conv2d_15_conv2d_readvariableop_resource:@@D
6sequential_3_conv2d_15_biasadd_readvariableop_resource:@G
3sequential_3_dense_6_matmul_readvariableop_resource:
? ?C
4sequential_3_dense_6_biasadd_readvariableop_resource:	?F
3sequential_3_dense_7_matmul_readvariableop_resource:	?
B
4sequential_3_dense_7_biasadd_readvariableop_resource:

identity??-sequential_3/conv2d_12/BiasAdd/ReadVariableOp?,sequential_3/conv2d_12/Conv2D/ReadVariableOp?-sequential_3/conv2d_13/BiasAdd/ReadVariableOp?,sequential_3/conv2d_13/Conv2D/ReadVariableOp?-sequential_3/conv2d_14/BiasAdd/ReadVariableOp?,sequential_3/conv2d_14/Conv2D/ReadVariableOp?-sequential_3/conv2d_15/BiasAdd/ReadVariableOp?,sequential_3/conv2d_15/Conv2D/ReadVariableOp?+sequential_3/dense_6/BiasAdd/ReadVariableOp?*sequential_3/dense_6/MatMul/ReadVariableOp?+sequential_3/dense_7/BiasAdd/ReadVariableOp?*sequential_3/dense_7/MatMul/ReadVariableOp?
,sequential_3/conv2d_12/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_3/conv2d_12/Conv2D/ReadVariableOp?
sequential_3/conv2d_12/Conv2DConv2Dinputs4sequential_3/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
sequential_3/conv2d_12/Conv2D?
-sequential_3/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_3/conv2d_12/BiasAdd/ReadVariableOp?
sequential_3/conv2d_12/BiasAddBiasAdd&sequential_3/conv2d_12/Conv2D:output:05sequential_3/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2 
sequential_3/conv2d_12/BiasAdd?
sequential_3/conv2d_12/ReluRelu'sequential_3/conv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:?????????   2
sequential_3/conv2d_12/Relu?
,sequential_3/conv2d_13/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02.
,sequential_3/conv2d_13/Conv2D/ReadVariableOp?
sequential_3/conv2d_13/Conv2DConv2D)sequential_3/conv2d_12/Relu:activations:04sequential_3/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
sequential_3/conv2d_13/Conv2D?
-sequential_3/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_3/conv2d_13/BiasAdd/ReadVariableOp?
sequential_3/conv2d_13/BiasAddBiasAdd&sequential_3/conv2d_13/Conv2D:output:05sequential_3/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2 
sequential_3/conv2d_13/BiasAdd?
sequential_3/conv2d_13/ReluRelu'sequential_3/conv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:?????????   2
sequential_3/conv2d_13/Relu?
$sequential_3/max_pooling2d_6/MaxPoolMaxPool)sequential_3/conv2d_13/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2&
$sequential_3/max_pooling2d_6/MaxPool?
sequential_3/dropout_9/IdentityIdentity-sequential_3/max_pooling2d_6/MaxPool:output:0*
T0*/
_output_shapes
:????????? 2!
sequential_3/dropout_9/Identity?
,sequential_3/conv2d_14/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02.
,sequential_3/conv2d_14/Conv2D/ReadVariableOp?
sequential_3/conv2d_14/Conv2DConv2D(sequential_3/dropout_9/Identity:output:04sequential_3/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
sequential_3/conv2d_14/Conv2D?
-sequential_3/conv2d_14/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_3/conv2d_14/BiasAdd/ReadVariableOp?
sequential_3/conv2d_14/BiasAddBiasAdd&sequential_3/conv2d_14/Conv2D:output:05sequential_3/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2 
sequential_3/conv2d_14/BiasAdd?
sequential_3/conv2d_14/ReluRelu'sequential_3/conv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
sequential_3/conv2d_14/Relu?
,sequential_3/conv2d_15/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,sequential_3/conv2d_15/Conv2D/ReadVariableOp?
sequential_3/conv2d_15/Conv2DConv2D)sequential_3/conv2d_14/Relu:activations:04sequential_3/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
sequential_3/conv2d_15/Conv2D?
-sequential_3/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_3/conv2d_15/BiasAdd/ReadVariableOp?
sequential_3/conv2d_15/BiasAddBiasAdd&sequential_3/conv2d_15/Conv2D:output:05sequential_3/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2 
sequential_3/conv2d_15/BiasAdd?
sequential_3/conv2d_15/ReluRelu'sequential_3/conv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
sequential_3/conv2d_15/Relu?
$sequential_3/max_pooling2d_7/MaxPoolMaxPool)sequential_3/conv2d_15/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2&
$sequential_3/max_pooling2d_7/MaxPool?
 sequential_3/dropout_10/IdentityIdentity-sequential_3/max_pooling2d_7/MaxPool:output:0*
T0*/
_output_shapes
:?????????@2"
 sequential_3/dropout_10/Identity?
sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
sequential_3/flatten_3/Const?
sequential_3/flatten_3/ReshapeReshape)sequential_3/dropout_10/Identity:output:0%sequential_3/flatten_3/Const:output:0*
T0*(
_output_shapes
:?????????? 2 
sequential_3/flatten_3/Reshape?
*sequential_3/dense_6/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype02,
*sequential_3/dense_6/MatMul/ReadVariableOp?
sequential_3/dense_6/MatMulMatMul'sequential_3/flatten_3/Reshape:output:02sequential_3/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_3/dense_6/MatMul?
+sequential_3/dense_6/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_3/dense_6/BiasAdd/ReadVariableOp?
sequential_3/dense_6/BiasAddBiasAdd%sequential_3/dense_6/MatMul:product:03sequential_3/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_3/dense_6/BiasAdd?
sequential_3/dense_6/ReluRelu%sequential_3/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_3/dense_6/Relu?
 sequential_3/dropout_11/IdentityIdentity'sequential_3/dense_6/Relu:activations:0*
T0*(
_output_shapes
:??????????2"
 sequential_3/dropout_11/Identity?
*sequential_3/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_7_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02,
*sequential_3/dense_7/MatMul/ReadVariableOp?
sequential_3/dense_7/MatMulMatMul)sequential_3/dropout_11/Identity:output:02sequential_3/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
sequential_3/dense_7/MatMul?
+sequential_3/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_7_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02-
+sequential_3/dense_7/BiasAdd/ReadVariableOp?
sequential_3/dense_7/BiasAddBiasAdd%sequential_3/dense_7/MatMul:product:03sequential_3/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
sequential_3/dense_7/BiasAdd?
sequential_3/dense_7/SoftmaxSoftmax%sequential_3/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
sequential_3/dense_7/Softmax?
IdentityIdentity&sequential_3/dense_7/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp.^sequential_3/conv2d_12/BiasAdd/ReadVariableOp-^sequential_3/conv2d_12/Conv2D/ReadVariableOp.^sequential_3/conv2d_13/BiasAdd/ReadVariableOp-^sequential_3/conv2d_13/Conv2D/ReadVariableOp.^sequential_3/conv2d_14/BiasAdd/ReadVariableOp-^sequential_3/conv2d_14/Conv2D/ReadVariableOp.^sequential_3/conv2d_15/BiasAdd/ReadVariableOp-^sequential_3/conv2d_15/Conv2D/ReadVariableOp,^sequential_3/dense_6/BiasAdd/ReadVariableOp+^sequential_3/dense_6/MatMul/ReadVariableOp,^sequential_3/dense_7/BiasAdd/ReadVariableOp+^sequential_3/dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????  : : : : : : : : : : : : 2^
-sequential_3/conv2d_12/BiasAdd/ReadVariableOp-sequential_3/conv2d_12/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_12/Conv2D/ReadVariableOp,sequential_3/conv2d_12/Conv2D/ReadVariableOp2^
-sequential_3/conv2d_13/BiasAdd/ReadVariableOp-sequential_3/conv2d_13/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_13/Conv2D/ReadVariableOp,sequential_3/conv2d_13/Conv2D/ReadVariableOp2^
-sequential_3/conv2d_14/BiasAdd/ReadVariableOp-sequential_3/conv2d_14/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_14/Conv2D/ReadVariableOp,sequential_3/conv2d_14/Conv2D/ReadVariableOp2^
-sequential_3/conv2d_15/BiasAdd/ReadVariableOp-sequential_3/conv2d_15/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_15/Conv2D/ReadVariableOp,sequential_3/conv2d_15/Conv2D/ReadVariableOp2Z
+sequential_3/dense_6/BiasAdd/ReadVariableOp+sequential_3/dense_6/BiasAdd/ReadVariableOp2X
*sequential_3/dense_6/MatMul/ReadVariableOp*sequential_3/dense_6/MatMul/ReadVariableOp2Z
+sequential_3/dense_7/BiasAdd/ReadVariableOp+sequential_3/dense_7/BiasAdd/ReadVariableOp2X
*sequential_3/dense_7/MatMul/ReadVariableOp*sequential_3/dense_7/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
(__inference_dense_7_layer_call_fn_124877

inputs
unknown:	?

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_1235982
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?c
?	
H__inference_sequential_3_layer_call_and_return_conditional_losses_124567

inputsB
(conv2d_12_conv2d_readvariableop_resource: 7
)conv2d_12_biasadd_readvariableop_resource: B
(conv2d_13_conv2d_readvariableop_resource:  7
)conv2d_13_biasadd_readvariableop_resource: B
(conv2d_14_conv2d_readvariableop_resource: @7
)conv2d_14_biasadd_readvariableop_resource:@B
(conv2d_15_conv2d_readvariableop_resource:@@7
)conv2d_15_biasadd_readvariableop_resource:@:
&dense_6_matmul_readvariableop_resource:
? ?6
'dense_6_biasadd_readvariableop_resource:	?9
&dense_7_matmul_readvariableop_resource:	?
5
'dense_7_biasadd_readvariableop_resource:

identity?? conv2d_12/BiasAdd/ReadVariableOp?conv2d_12/Conv2D/ReadVariableOp? conv2d_13/BiasAdd/ReadVariableOp?conv2d_13/Conv2D/ReadVariableOp? conv2d_14/BiasAdd/ReadVariableOp?conv2d_14/Conv2D/ReadVariableOp? conv2d_15/BiasAdd/ReadVariableOp?conv2d_15/Conv2D/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?dense_7/BiasAdd/ReadVariableOp?dense_7/MatMul/ReadVariableOp?
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_12/Conv2D/ReadVariableOp?
conv2d_12/Conv2DConv2Dinputs'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
conv2d_12/Conv2D?
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_12/BiasAdd/ReadVariableOp?
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
conv2d_12/BiasAdd~
conv2d_12/ReluReluconv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:?????????   2
conv2d_12/Relu?
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_13/Conv2D/ReadVariableOp?
conv2d_13/Conv2DConv2Dconv2d_12/Relu:activations:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
conv2d_13/Conv2D?
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_13/BiasAdd/ReadVariableOp?
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
conv2d_13/BiasAdd~
conv2d_13/ReluReluconv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:?????????   2
conv2d_13/Relu?
max_pooling2d_6/MaxPoolMaxPoolconv2d_13/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_6/MaxPoolw
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_9/dropout/Const?
dropout_9/dropout/MulMul max_pooling2d_6/MaxPool:output:0 dropout_9/dropout/Const:output:0*
T0*/
_output_shapes
:????????? 2
dropout_9/dropout/Mul?
dropout_9/dropout/ShapeShape max_pooling2d_6/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_9/dropout/Shape?
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*/
_output_shapes
:????????? *
dtype020
.dropout_9/dropout/random_uniform/RandomUniform?
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2"
 dropout_9/dropout/GreaterEqual/y?
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? 2 
dropout_9/dropout/GreaterEqual?
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? 2
dropout_9/dropout/Cast?
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*/
_output_shapes
:????????? 2
dropout_9/dropout/Mul_1?
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_14/Conv2D/ReadVariableOp?
conv2d_14/Conv2DConv2Ddropout_9/dropout/Mul_1:z:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_14/Conv2D?
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_14/BiasAdd/ReadVariableOp?
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_14/BiasAdd~
conv2d_14/ReluReluconv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_14/Relu?
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_15/Conv2D/ReadVariableOp?
conv2d_15/Conv2DConv2Dconv2d_14/Relu:activations:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_15/Conv2D?
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_15/BiasAdd/ReadVariableOp?
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_15/BiasAdd~
conv2d_15/ReluReluconv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_15/Relu?
max_pooling2d_7/MaxPoolMaxPoolconv2d_15/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_7/MaxPooly
dropout_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_10/dropout/Const?
dropout_10/dropout/MulMul max_pooling2d_7/MaxPool:output:0!dropout_10/dropout/Const:output:0*
T0*/
_output_shapes
:?????????@2
dropout_10/dropout/Mul?
dropout_10/dropout/ShapeShape max_pooling2d_7/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_10/dropout/Shape?
/dropout_10/dropout/random_uniform/RandomUniformRandomUniform!dropout_10/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype021
/dropout_10/dropout/random_uniform/RandomUniform?
!dropout_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2#
!dropout_10/dropout/GreaterEqual/y?
dropout_10/dropout/GreaterEqualGreaterEqual8dropout_10/dropout/random_uniform/RandomUniform:output:0*dropout_10/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2!
dropout_10/dropout/GreaterEqual?
dropout_10/dropout/CastCast#dropout_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2
dropout_10/dropout/Cast?
dropout_10/dropout/Mul_1Muldropout_10/dropout/Mul:z:0dropout_10/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2
dropout_10/dropout/Mul_1s
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_3/Const?
flatten_3/ReshapeReshapedropout_10/dropout/Mul_1:z:0flatten_3/Const:output:0*
T0*(
_output_shapes
:?????????? 2
flatten_3/Reshape?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMulflatten_3/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_6/BiasAddq
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_6/Reluy
dropout_11/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_11/dropout/Const?
dropout_11/dropout/MulMuldense_6/Relu:activations:0!dropout_11/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_11/dropout/Mul~
dropout_11/dropout/ShapeShapedense_6/Relu:activations:0*
T0*
_output_shapes
:2
dropout_11/dropout/Shape?
/dropout_11/dropout/random_uniform/RandomUniformRandomUniform!dropout_11/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype021
/dropout_11/dropout/random_uniform/RandomUniform?
!dropout_11/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_11/dropout/GreaterEqual/y?
dropout_11/dropout/GreaterEqualGreaterEqual8dropout_11/dropout/random_uniform/RandomUniform:output:0*dropout_11/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2!
dropout_11/dropout/GreaterEqual?
dropout_11/dropout/CastCast#dropout_11/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_11/dropout/Cast?
dropout_11/dropout/Mul_1Muldropout_11/dropout/Mul:z:0dropout_11/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_11/dropout/Mul_1?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMuldropout_11/dropout/Mul_1:z:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_7/MatMul?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_7/BiasAddy
dense_7/SoftmaxSoftmaxdense_7/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_7/Softmaxt
IdentityIdentitydense_7/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????  : : : : : : : : : : : : 2D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
-__inference_sequential_3_layer_call_fn_123904
conv2d_12_input!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:
? ?
	unknown_8:	?
	unknown_9:	?


unknown_10:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_1238482
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????  : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????  
)
_user_specified_nameconv2d_12_input
?b
?
__inference__traced_save_125035
file_prefix/
+savev2_conv2d_12_kernel_read_readvariableop-
)savev2_conv2d_12_bias_read_readvariableop/
+savev2_conv2d_13_kernel_read_readvariableop-
)savev2_conv2d_13_bias_read_readvariableop/
+savev2_conv2d_14_kernel_read_readvariableop-
)savev2_conv2d_14_bias_read_readvariableop/
+savev2_conv2d_15_kernel_read_readvariableop-
)savev2_conv2d_15_bias_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv2d_12_kernel_m_read_readvariableop4
0savev2_adam_conv2d_12_bias_m_read_readvariableop6
2savev2_adam_conv2d_13_kernel_m_read_readvariableop4
0savev2_adam_conv2d_13_bias_m_read_readvariableop6
2savev2_adam_conv2d_14_kernel_m_read_readvariableop4
0savev2_adam_conv2d_14_bias_m_read_readvariableop6
2savev2_adam_conv2d_15_kernel_m_read_readvariableop4
0savev2_adam_conv2d_15_bias_m_read_readvariableop4
0savev2_adam_dense_6_kernel_m_read_readvariableop2
.savev2_adam_dense_6_bias_m_read_readvariableop4
0savev2_adam_dense_7_kernel_m_read_readvariableop2
.savev2_adam_dense_7_bias_m_read_readvariableop6
2savev2_adam_conv2d_12_kernel_v_read_readvariableop4
0savev2_adam_conv2d_12_bias_v_read_readvariableop6
2savev2_adam_conv2d_13_kernel_v_read_readvariableop4
0savev2_adam_conv2d_13_bias_v_read_readvariableop6
2savev2_adam_conv2d_14_kernel_v_read_readvariableop4
0savev2_adam_conv2d_14_bias_v_read_readvariableop6
2savev2_adam_conv2d_15_kernel_v_read_readvariableop4
0savev2_adam_conv2d_15_bias_v_read_readvariableop4
0savev2_adam_dense_6_kernel_v_read_readvariableop2
.savev2_adam_dense_6_bias_v_read_readvariableop4
0savev2_adam_dense_7_kernel_v_read_readvariableop2
.savev2_adam_dense_7_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
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
Const_1?
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
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*?
value?B?.B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-0/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-0/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-0/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-0/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-0/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-0/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-0/keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-0/keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/0/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/1/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/2/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/3/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/4/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/5/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/6/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/7/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/8/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/9/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBbtrainable_variables/10/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBbtrainable_variables/11/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/0/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/1/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/2/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/3/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/4/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/5/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/6/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/7/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/8/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/9/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBbtrainable_variables/10/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBbtrainable_variables/11/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_12_kernel_read_readvariableop)savev2_conv2d_12_bias_read_readvariableop+savev2_conv2d_13_kernel_read_readvariableop)savev2_conv2d_13_bias_read_readvariableop+savev2_conv2d_14_kernel_read_readvariableop)savev2_conv2d_14_bias_read_readvariableop+savev2_conv2d_15_kernel_read_readvariableop)savev2_conv2d_15_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv2d_12_kernel_m_read_readvariableop0savev2_adam_conv2d_12_bias_m_read_readvariableop2savev2_adam_conv2d_13_kernel_m_read_readvariableop0savev2_adam_conv2d_13_bias_m_read_readvariableop2savev2_adam_conv2d_14_kernel_m_read_readvariableop0savev2_adam_conv2d_14_bias_m_read_readvariableop2savev2_adam_conv2d_15_kernel_m_read_readvariableop0savev2_adam_conv2d_15_bias_m_read_readvariableop0savev2_adam_dense_6_kernel_m_read_readvariableop.savev2_adam_dense_6_bias_m_read_readvariableop0savev2_adam_dense_7_kernel_m_read_readvariableop.savev2_adam_dense_7_bias_m_read_readvariableop2savev2_adam_conv2d_12_kernel_v_read_readvariableop0savev2_adam_conv2d_12_bias_v_read_readvariableop2savev2_adam_conv2d_13_kernel_v_read_readvariableop0savev2_adam_conv2d_13_bias_v_read_readvariableop2savev2_adam_conv2d_14_kernel_v_read_readvariableop0savev2_adam_conv2d_14_bias_v_read_readvariableop2savev2_adam_conv2d_15_kernel_v_read_readvariableop0savev2_adam_conv2d_15_bias_v_read_readvariableop0savev2_adam_dense_6_kernel_v_read_readvariableop.savev2_adam_dense_6_bias_v_read_readvariableop0savev2_adam_dense_7_kernel_v_read_readvariableop.savev2_adam_dense_7_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *<
dtypes2
02.	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: : : :  : : @:@:@@:@:
? ?:?:	?
:
: : : : : : : : : : : :  : : @:@:@@:@:
? ?:?:	?
:
: : :  : : @:@:@@:@:
? ?:?:	?
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
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:&	"
 
_output_shapes
:
? ?:!


_output_shapes	
:?:%!

_output_shapes
:	?
: 

_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:&"
 
_output_shapes
:
? ?:!

_output_shapes	
:?:% !

_output_shapes
:	?
: !

_output_shapes
:
:,"(
&
_output_shapes
: : #

_output_shapes
: :,$(
&
_output_shapes
:  : %

_output_shapes
: :,&(
&
_output_shapes
: @: '

_output_shapes
:@:,((
&
_output_shapes
:@@: )

_output_shapes
:@:&*"
 
_output_shapes
:
? ?:!+

_output_shapes	
:?:%,!

_output_shapes
:	?
: -

_output_shapes
:
:.

_output_shapes
: 
?
d
F__inference_dropout_11_layer_call_and_return_conditional_losses_123585

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
F__inference_dropout_10_layer_call_and_return_conditional_losses_123701

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
d
F__inference_dropout_10_layer_call_and_return_conditional_losses_124777

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
C__inference_dense_6_layer_call_and_return_conditional_losses_124821

inputs2
matmul_readvariableop_resource:
? ?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_123419

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
C__inference_dense_7_layer_call_and_return_conditional_losses_123598

inputs1
matmul_readvariableop_resource:	?
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_124675

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????   :W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
c
E__inference_dropout_9_layer_call_and_return_conditional_losses_124690

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:????????? 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:????????? 2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
*__inference_conv2d_13_layer_call_fn_124665

inputs!
unknown:  
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_1234892
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????   2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????   : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?5
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_123944
conv2d_12_input*
conv2d_12_123907: 
conv2d_12_123909: *
conv2d_13_123912:  
conv2d_13_123914: *
conv2d_14_123919: @
conv2d_14_123921:@*
conv2d_15_123924:@@
conv2d_15_123926:@"
dense_6_123932:
? ?
dense_6_123934:	?!
dense_7_123938:	?

dense_7_123940:

identity??!conv2d_12/StatefulPartitionedCall?!conv2d_13/StatefulPartitionedCall?!conv2d_14/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCallconv2d_12_inputconv2d_12_123907conv2d_12_123909*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_1234722#
!conv2d_12/StatefulPartitionedCall?
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_123912conv2d_13_123914*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_1234892#
!conv2d_13/StatefulPartitionedCall?
max_pooling2d_6/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_1234992!
max_pooling2d_6/PartitionedCall?
dropout_9/PartitionedCallPartitionedCall(max_pooling2d_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_1235062
dropout_9/PartitionedCall?
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall"dropout_9/PartitionedCall:output:0conv2d_14_123919conv2d_14_123921*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_1235192#
!conv2d_14/StatefulPartitionedCall?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0conv2d_15_123924conv2d_15_123926*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_1235362#
!conv2d_15/StatefulPartitionedCall?
max_pooling2d_7/PartitionedCallPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_1235462!
max_pooling2d_7/PartitionedCall?
dropout_10/PartitionedCallPartitionedCall(max_pooling2d_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_1235532
dropout_10/PartitionedCall?
flatten_3/PartitionedCallPartitionedCall#dropout_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_1235612
flatten_3/PartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_6_123932dense_6_123934*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_1235742!
dense_6/StatefulPartitionedCall?
dropout_11/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_1235852
dropout_11/PartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0dense_7_123938dense_7_123940*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_1235982!
dense_7/StatefulPartitionedCall?
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????  : : : : : : : : : : : : 2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:` \
/
_output_shapes
:?????????  
)
_user_specified_nameconv2d_12_input
?
?
-__inference_sequential_3_layer_call_fn_124625

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:
? ?
	unknown_8:	?
	unknown_9:	?


unknown_10:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_1238482
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????  : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_123441

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_124762

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
L
0__inference_max_pooling2d_7_layer_call_fn_124772

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_1235462
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
(__inference_dense_6_layer_call_fn_124830

inputs
unknown:
? ?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_1235742
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?}
?
D__inference_model_45_layer_call_and_return_conditional_losses_124376

inputsO
5sequential_3_conv2d_12_conv2d_readvariableop_resource: D
6sequential_3_conv2d_12_biasadd_readvariableop_resource: O
5sequential_3_conv2d_13_conv2d_readvariableop_resource:  D
6sequential_3_conv2d_13_biasadd_readvariableop_resource: O
5sequential_3_conv2d_14_conv2d_readvariableop_resource: @D
6sequential_3_conv2d_14_biasadd_readvariableop_resource:@O
5sequential_3_conv2d_15_conv2d_readvariableop_resource:@@D
6sequential_3_conv2d_15_biasadd_readvariableop_resource:@G
3sequential_3_dense_6_matmul_readvariableop_resource:
? ?C
4sequential_3_dense_6_biasadd_readvariableop_resource:	?F
3sequential_3_dense_7_matmul_readvariableop_resource:	?
B
4sequential_3_dense_7_biasadd_readvariableop_resource:

identity??-sequential_3/conv2d_12/BiasAdd/ReadVariableOp?,sequential_3/conv2d_12/Conv2D/ReadVariableOp?-sequential_3/conv2d_13/BiasAdd/ReadVariableOp?,sequential_3/conv2d_13/Conv2D/ReadVariableOp?-sequential_3/conv2d_14/BiasAdd/ReadVariableOp?,sequential_3/conv2d_14/Conv2D/ReadVariableOp?-sequential_3/conv2d_15/BiasAdd/ReadVariableOp?,sequential_3/conv2d_15/Conv2D/ReadVariableOp?+sequential_3/dense_6/BiasAdd/ReadVariableOp?*sequential_3/dense_6/MatMul/ReadVariableOp?+sequential_3/dense_7/BiasAdd/ReadVariableOp?*sequential_3/dense_7/MatMul/ReadVariableOp?
,sequential_3/conv2d_12/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_3/conv2d_12/Conv2D/ReadVariableOp?
sequential_3/conv2d_12/Conv2DConv2Dinputs4sequential_3/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
sequential_3/conv2d_12/Conv2D?
-sequential_3/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_3/conv2d_12/BiasAdd/ReadVariableOp?
sequential_3/conv2d_12/BiasAddBiasAdd&sequential_3/conv2d_12/Conv2D:output:05sequential_3/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2 
sequential_3/conv2d_12/BiasAdd?
sequential_3/conv2d_12/ReluRelu'sequential_3/conv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:?????????   2
sequential_3/conv2d_12/Relu?
,sequential_3/conv2d_13/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02.
,sequential_3/conv2d_13/Conv2D/ReadVariableOp?
sequential_3/conv2d_13/Conv2DConv2D)sequential_3/conv2d_12/Relu:activations:04sequential_3/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
sequential_3/conv2d_13/Conv2D?
-sequential_3/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_3/conv2d_13/BiasAdd/ReadVariableOp?
sequential_3/conv2d_13/BiasAddBiasAdd&sequential_3/conv2d_13/Conv2D:output:05sequential_3/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2 
sequential_3/conv2d_13/BiasAdd?
sequential_3/conv2d_13/ReluRelu'sequential_3/conv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:?????????   2
sequential_3/conv2d_13/Relu?
$sequential_3/max_pooling2d_6/MaxPoolMaxPool)sequential_3/conv2d_13/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2&
$sequential_3/max_pooling2d_6/MaxPool?
$sequential_3/dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2&
$sequential_3/dropout_9/dropout/Const?
"sequential_3/dropout_9/dropout/MulMul-sequential_3/max_pooling2d_6/MaxPool:output:0-sequential_3/dropout_9/dropout/Const:output:0*
T0*/
_output_shapes
:????????? 2$
"sequential_3/dropout_9/dropout/Mul?
$sequential_3/dropout_9/dropout/ShapeShape-sequential_3/max_pooling2d_6/MaxPool:output:0*
T0*
_output_shapes
:2&
$sequential_3/dropout_9/dropout/Shape?
;sequential_3/dropout_9/dropout/random_uniform/RandomUniformRandomUniform-sequential_3/dropout_9/dropout/Shape:output:0*
T0*/
_output_shapes
:????????? *
dtype02=
;sequential_3/dropout_9/dropout/random_uniform/RandomUniform?
-sequential_3/dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2/
-sequential_3/dropout_9/dropout/GreaterEqual/y?
+sequential_3/dropout_9/dropout/GreaterEqualGreaterEqualDsequential_3/dropout_9/dropout/random_uniform/RandomUniform:output:06sequential_3/dropout_9/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? 2-
+sequential_3/dropout_9/dropout/GreaterEqual?
#sequential_3/dropout_9/dropout/CastCast/sequential_3/dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? 2%
#sequential_3/dropout_9/dropout/Cast?
$sequential_3/dropout_9/dropout/Mul_1Mul&sequential_3/dropout_9/dropout/Mul:z:0'sequential_3/dropout_9/dropout/Cast:y:0*
T0*/
_output_shapes
:????????? 2&
$sequential_3/dropout_9/dropout/Mul_1?
,sequential_3/conv2d_14/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02.
,sequential_3/conv2d_14/Conv2D/ReadVariableOp?
sequential_3/conv2d_14/Conv2DConv2D(sequential_3/dropout_9/dropout/Mul_1:z:04sequential_3/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
sequential_3/conv2d_14/Conv2D?
-sequential_3/conv2d_14/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_3/conv2d_14/BiasAdd/ReadVariableOp?
sequential_3/conv2d_14/BiasAddBiasAdd&sequential_3/conv2d_14/Conv2D:output:05sequential_3/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2 
sequential_3/conv2d_14/BiasAdd?
sequential_3/conv2d_14/ReluRelu'sequential_3/conv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
sequential_3/conv2d_14/Relu?
,sequential_3/conv2d_15/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,sequential_3/conv2d_15/Conv2D/ReadVariableOp?
sequential_3/conv2d_15/Conv2DConv2D)sequential_3/conv2d_14/Relu:activations:04sequential_3/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
sequential_3/conv2d_15/Conv2D?
-sequential_3/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_3/conv2d_15/BiasAdd/ReadVariableOp?
sequential_3/conv2d_15/BiasAddBiasAdd&sequential_3/conv2d_15/Conv2D:output:05sequential_3/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2 
sequential_3/conv2d_15/BiasAdd?
sequential_3/conv2d_15/ReluRelu'sequential_3/conv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
sequential_3/conv2d_15/Relu?
$sequential_3/max_pooling2d_7/MaxPoolMaxPool)sequential_3/conv2d_15/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2&
$sequential_3/max_pooling2d_7/MaxPool?
%sequential_3/dropout_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2'
%sequential_3/dropout_10/dropout/Const?
#sequential_3/dropout_10/dropout/MulMul-sequential_3/max_pooling2d_7/MaxPool:output:0.sequential_3/dropout_10/dropout/Const:output:0*
T0*/
_output_shapes
:?????????@2%
#sequential_3/dropout_10/dropout/Mul?
%sequential_3/dropout_10/dropout/ShapeShape-sequential_3/max_pooling2d_7/MaxPool:output:0*
T0*
_output_shapes
:2'
%sequential_3/dropout_10/dropout/Shape?
<sequential_3/dropout_10/dropout/random_uniform/RandomUniformRandomUniform.sequential_3/dropout_10/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype02>
<sequential_3/dropout_10/dropout/random_uniform/RandomUniform?
.sequential_3/dropout_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>20
.sequential_3/dropout_10/dropout/GreaterEqual/y?
,sequential_3/dropout_10/dropout/GreaterEqualGreaterEqualEsequential_3/dropout_10/dropout/random_uniform/RandomUniform:output:07sequential_3/dropout_10/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2.
,sequential_3/dropout_10/dropout/GreaterEqual?
$sequential_3/dropout_10/dropout/CastCast0sequential_3/dropout_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2&
$sequential_3/dropout_10/dropout/Cast?
%sequential_3/dropout_10/dropout/Mul_1Mul'sequential_3/dropout_10/dropout/Mul:z:0(sequential_3/dropout_10/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2'
%sequential_3/dropout_10/dropout/Mul_1?
sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
sequential_3/flatten_3/Const?
sequential_3/flatten_3/ReshapeReshape)sequential_3/dropout_10/dropout/Mul_1:z:0%sequential_3/flatten_3/Const:output:0*
T0*(
_output_shapes
:?????????? 2 
sequential_3/flatten_3/Reshape?
*sequential_3/dense_6/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype02,
*sequential_3/dense_6/MatMul/ReadVariableOp?
sequential_3/dense_6/MatMulMatMul'sequential_3/flatten_3/Reshape:output:02sequential_3/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_3/dense_6/MatMul?
+sequential_3/dense_6/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_3/dense_6/BiasAdd/ReadVariableOp?
sequential_3/dense_6/BiasAddBiasAdd%sequential_3/dense_6/MatMul:product:03sequential_3/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_3/dense_6/BiasAdd?
sequential_3/dense_6/ReluRelu%sequential_3/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_3/dense_6/Relu?
%sequential_3/dropout_11/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_3/dropout_11/dropout/Const?
#sequential_3/dropout_11/dropout/MulMul'sequential_3/dense_6/Relu:activations:0.sequential_3/dropout_11/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2%
#sequential_3/dropout_11/dropout/Mul?
%sequential_3/dropout_11/dropout/ShapeShape'sequential_3/dense_6/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_3/dropout_11/dropout/Shape?
<sequential_3/dropout_11/dropout/random_uniform/RandomUniformRandomUniform.sequential_3/dropout_11/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02>
<sequential_3/dropout_11/dropout/random_uniform/RandomUniform?
.sequential_3/dropout_11/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_3/dropout_11/dropout/GreaterEqual/y?
,sequential_3/dropout_11/dropout/GreaterEqualGreaterEqualEsequential_3/dropout_11/dropout/random_uniform/RandomUniform:output:07sequential_3/dropout_11/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2.
,sequential_3/dropout_11/dropout/GreaterEqual?
$sequential_3/dropout_11/dropout/CastCast0sequential_3/dropout_11/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2&
$sequential_3/dropout_11/dropout/Cast?
%sequential_3/dropout_11/dropout/Mul_1Mul'sequential_3/dropout_11/dropout/Mul:z:0(sequential_3/dropout_11/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2'
%sequential_3/dropout_11/dropout/Mul_1?
*sequential_3/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_7_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02,
*sequential_3/dense_7/MatMul/ReadVariableOp?
sequential_3/dense_7/MatMulMatMul)sequential_3/dropout_11/dropout/Mul_1:z:02sequential_3/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
sequential_3/dense_7/MatMul?
+sequential_3/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_7_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02-
+sequential_3/dense_7/BiasAdd/ReadVariableOp?
sequential_3/dense_7/BiasAddBiasAdd%sequential_3/dense_7/MatMul:product:03sequential_3/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
sequential_3/dense_7/BiasAdd?
sequential_3/dense_7/SoftmaxSoftmax%sequential_3/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
sequential_3/dense_7/Softmax?
IdentityIdentity&sequential_3/dense_7/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp.^sequential_3/conv2d_12/BiasAdd/ReadVariableOp-^sequential_3/conv2d_12/Conv2D/ReadVariableOp.^sequential_3/conv2d_13/BiasAdd/ReadVariableOp-^sequential_3/conv2d_13/Conv2D/ReadVariableOp.^sequential_3/conv2d_14/BiasAdd/ReadVariableOp-^sequential_3/conv2d_14/Conv2D/ReadVariableOp.^sequential_3/conv2d_15/BiasAdd/ReadVariableOp-^sequential_3/conv2d_15/Conv2D/ReadVariableOp,^sequential_3/dense_6/BiasAdd/ReadVariableOp+^sequential_3/dense_6/MatMul/ReadVariableOp,^sequential_3/dense_7/BiasAdd/ReadVariableOp+^sequential_3/dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????  : : : : : : : : : : : : 2^
-sequential_3/conv2d_12/BiasAdd/ReadVariableOp-sequential_3/conv2d_12/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_12/Conv2D/ReadVariableOp,sequential_3/conv2d_12/Conv2D/ReadVariableOp2^
-sequential_3/conv2d_13/BiasAdd/ReadVariableOp-sequential_3/conv2d_13/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_13/Conv2D/ReadVariableOp,sequential_3/conv2d_13/Conv2D/ReadVariableOp2^
-sequential_3/conv2d_14/BiasAdd/ReadVariableOp-sequential_3/conv2d_14/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_14/Conv2D/ReadVariableOp,sequential_3/conv2d_14/Conv2D/ReadVariableOp2^
-sequential_3/conv2d_15/BiasAdd/ReadVariableOp-sequential_3/conv2d_15/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_15/Conv2D/ReadVariableOp,sequential_3/conv2d_15/Conv2D/ReadVariableOp2Z
+sequential_3/dense_6/BiasAdd/ReadVariableOp+sequential_3/dense_6/BiasAdd/ReadVariableOp2X
*sequential_3/dense_6/MatMul/ReadVariableOp*sequential_3/dense_6/MatMul/ReadVariableOp2Z
+sequential_3/dense_7/BiasAdd/ReadVariableOp+sequential_3/dense_7/BiasAdd/ReadVariableOp2X
*sequential_3/dense_7/MatMul/ReadVariableOp*sequential_3/dense_7/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
G
+__inference_dropout_11_layer_call_fn_124852

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_1235852
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
?
input6
serving_default_input:0?????????  @
sequential_30
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
trainable_variables
	variables
regularization_losses
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?__call__
?_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
?
layer_with_weights-0
layer-0
	layer_with_weights-1
	layer-1

layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
layer-8
layer_with_weights-4
layer-9
layer-10
layer_with_weights-5
layer-11
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_sequential
v
0
1
2
3
4
5
6
 7
!8
"9
#10
$11"
trackable_list_wrapper
v
0
1
2
3
4
5
6
 7
!8
"9
#10
$11"
trackable_list_wrapper
 "
trackable_list_wrapper
?
%layer_metrics
&non_trainable_variables
trainable_variables

'layers
	variables
(layer_regularization_losses
)metrics
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?

kernel
bias
*trainable_variables
+	variables
,regularization_losses
-	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

kernel
bias
.trainable_variables
/	variables
0regularization_losses
1	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
2trainable_variables
3	variables
4regularization_losses
5	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
6trainable_variables
7	variables
8regularization_losses
9	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

kernel
bias
:trainable_variables
;	variables
<regularization_losses
=	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

kernel
 bias
>trainable_variables
?	variables
@regularization_losses
A	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
Btrainable_variables
C	variables
Dregularization_losses
E	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
Ftrainable_variables
G	variables
Hregularization_losses
I	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
Jtrainable_variables
K	variables
Lregularization_losses
M	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

!kernel
"bias
Ntrainable_variables
O	variables
Pregularization_losses
Q	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
Rtrainable_variables
S	variables
Tregularization_losses
U	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

#kernel
$bias
Vtrainable_variables
W	variables
Xregularization_losses
Y	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
Ziter

[beta_1

\beta_2
	]decay
^learning_ratem?m?m?m?m?m?m? m?!m?"m?#m?$m?v?v?v?v?v?v?v? v?!v?"v?#v?$v?"
	optimizer
v
0
1
2
3
4
5
6
 7
!8
"9
#10
$11"
trackable_list_wrapper
v
0
1
2
3
4
5
6
 7
!8
"9
#10
$11"
trackable_list_wrapper
 "
trackable_list_wrapper
?
_layer_metrics
`non_trainable_variables
trainable_variables

alayers
	variables
blayer_regularization_losses
cmetrics
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:( 2conv2d_12/kernel
: 2conv2d_12/bias
*:(  2conv2d_13/kernel
: 2conv2d_13/bias
*:( @2conv2d_14/kernel
:@2conv2d_14/bias
*:(@@2conv2d_15/kernel
:@2conv2d_15/bias
": 
? ?2dense_6/kernel
:?2dense_6/bias
!:	?
2dense_7/kernel
:
2dense_7/bias
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
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
dlayer_metrics

elayers
fnon_trainable_variables
*trainable_variables
+	variables
glayer_regularization_losses
,regularization_losses
hmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
ilayer_metrics

jlayers
knon_trainable_variables
.trainable_variables
/	variables
llayer_regularization_losses
0regularization_losses
mmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
nlayer_metrics

olayers
pnon_trainable_variables
2trainable_variables
3	variables
qlayer_regularization_losses
4regularization_losses
rmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
slayer_metrics

tlayers
unon_trainable_variables
6trainable_variables
7	variables
vlayer_regularization_losses
8regularization_losses
wmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
xlayer_metrics

ylayers
znon_trainable_variables
:trainable_variables
;	variables
{layer_regularization_losses
<regularization_losses
|metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
}layer_metrics

~layers
non_trainable_variables
>trainable_variables
?	variables
 ?layer_regularization_losses
@regularization_losses
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?layers
?non_trainable_variables
Btrainable_variables
C	variables
 ?layer_regularization_losses
Dregularization_losses
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?layers
?non_trainable_variables
Ftrainable_variables
G	variables
 ?layer_regularization_losses
Hregularization_losses
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?layers
?non_trainable_variables
Jtrainable_variables
K	variables
 ?layer_regularization_losses
Lregularization_losses
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?layers
?non_trainable_variables
Ntrainable_variables
O	variables
 ?layer_regularization_losses
Pregularization_losses
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?layers
?non_trainable_variables
Rtrainable_variables
S	variables
 ?layer_regularization_losses
Tregularization_losses
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?layers
?non_trainable_variables
Vtrainable_variables
W	variables
 ?layer_regularization_losses
Xregularization_losses
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
v
0
	1

2
3
4
5
6
7
8
9
10
11"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
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
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
/:- 2Adam/conv2d_12/kernel/m
!: 2Adam/conv2d_12/bias/m
/:-  2Adam/conv2d_13/kernel/m
!: 2Adam/conv2d_13/bias/m
/:- @2Adam/conv2d_14/kernel/m
!:@2Adam/conv2d_14/bias/m
/:-@@2Adam/conv2d_15/kernel/m
!:@2Adam/conv2d_15/bias/m
':%
? ?2Adam/dense_6/kernel/m
 :?2Adam/dense_6/bias/m
&:$	?
2Adam/dense_7/kernel/m
:
2Adam/dense_7/bias/m
/:- 2Adam/conv2d_12/kernel/v
!: 2Adam/conv2d_12/bias/v
/:-  2Adam/conv2d_13/kernel/v
!: 2Adam/conv2d_13/bias/v
/:- @2Adam/conv2d_14/kernel/v
!:@2Adam/conv2d_14/bias/v
/:-@@2Adam/conv2d_15/kernel/v
!:@2Adam/conv2d_15/bias/v
':%
? ?2Adam/dense_6/kernel/v
 :?2Adam/dense_6/bias/v
&:$	?
2Adam/dense_7/kernel/v
:
2Adam/dense_7/bias/v
?2?
D__inference_model_45_layer_call_and_return_conditional_losses_124302
D__inference_model_45_layer_call_and_return_conditional_losses_124376
D__inference_model_45_layer_call_and_return_conditional_losses_124189
D__inference_model_45_layer_call_and_return_conditional_losses_124218?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_model_45_layer_call_fn_124044
)__inference_model_45_layer_call_fn_124405
)__inference_model_45_layer_call_fn_124434
)__inference_model_45_layer_call_fn_124160?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
!__inference__wrapped_model_123410input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_sequential_3_layer_call_and_return_conditional_losses_124493
H__inference_sequential_3_layer_call_and_return_conditional_losses_124567
H__inference_sequential_3_layer_call_and_return_conditional_losses_123944
H__inference_sequential_3_layer_call_and_return_conditional_losses_123984?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_sequential_3_layer_call_fn_123632
-__inference_sequential_3_layer_call_fn_124596
-__inference_sequential_3_layer_call_fn_124625
-__inference_sequential_3_layer_call_fn_123904?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
$__inference_signature_wrapper_124249input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_12_layer_call_and_return_conditional_losses_124636?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_12_layer_call_fn_124645?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_13_layer_call_and_return_conditional_losses_124656?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_13_layer_call_fn_124665?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_124670
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_124675?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_max_pooling2d_6_layer_call_fn_124680
0__inference_max_pooling2d_6_layer_call_fn_124685?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dropout_9_layer_call_and_return_conditional_losses_124690
E__inference_dropout_9_layer_call_and_return_conditional_losses_124702?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_dropout_9_layer_call_fn_124707
*__inference_dropout_9_layer_call_fn_124712?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_conv2d_14_layer_call_and_return_conditional_losses_124723?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_14_layer_call_fn_124732?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_15_layer_call_and_return_conditional_losses_124743?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_15_layer_call_fn_124752?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_124757
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_124762?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_max_pooling2d_7_layer_call_fn_124767
0__inference_max_pooling2d_7_layer_call_fn_124772?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dropout_10_layer_call_and_return_conditional_losses_124777
F__inference_dropout_10_layer_call_and_return_conditional_losses_124789?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_dropout_10_layer_call_fn_124794
+__inference_dropout_10_layer_call_fn_124799?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_flatten_3_layer_call_and_return_conditional_losses_124805?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_flatten_3_layer_call_fn_124810?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_6_layer_call_and_return_conditional_losses_124821?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_6_layer_call_fn_124830?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dropout_11_layer_call_and_return_conditional_losses_124835
F__inference_dropout_11_layer_call_and_return_conditional_losses_124847?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_dropout_11_layer_call_fn_124852
+__inference_dropout_11_layer_call_fn_124857?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_dense_7_layer_call_and_return_conditional_losses_124868?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_7_layer_call_fn_124877?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
!__inference__wrapped_model_123410? !"#$6?3
,?)
'?$
input?????????  
? ";?8
6
sequential_3&?#
sequential_3?????????
?
E__inference_conv2d_12_layer_call_and_return_conditional_losses_124636l7?4
-?*
(?%
inputs?????????  
? "-?*
#? 
0?????????   
? ?
*__inference_conv2d_12_layer_call_fn_124645_7?4
-?*
(?%
inputs?????????  
? " ??????????   ?
E__inference_conv2d_13_layer_call_and_return_conditional_losses_124656l7?4
-?*
(?%
inputs?????????   
? "-?*
#? 
0?????????   
? ?
*__inference_conv2d_13_layer_call_fn_124665_7?4
-?*
(?%
inputs?????????   
? " ??????????   ?
E__inference_conv2d_14_layer_call_and_return_conditional_losses_124723l7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????@
? ?
*__inference_conv2d_14_layer_call_fn_124732_7?4
-?*
(?%
inputs????????? 
? " ??????????@?
E__inference_conv2d_15_layer_call_and_return_conditional_losses_124743l 7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????@
? ?
*__inference_conv2d_15_layer_call_fn_124752_ 7?4
-?*
(?%
inputs?????????@
? " ??????????@?
C__inference_dense_6_layer_call_and_return_conditional_losses_124821^!"0?-
&?#
!?
inputs?????????? 
? "&?#
?
0??????????
? }
(__inference_dense_6_layer_call_fn_124830Q!"0?-
&?#
!?
inputs?????????? 
? "????????????
C__inference_dense_7_layer_call_and_return_conditional_losses_124868]#$0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????

? |
(__inference_dense_7_layer_call_fn_124877P#$0?-
&?#
!?
inputs??????????
? "??????????
?
F__inference_dropout_10_layer_call_and_return_conditional_losses_124777l;?8
1?.
(?%
inputs?????????@
p 
? "-?*
#? 
0?????????@
? ?
F__inference_dropout_10_layer_call_and_return_conditional_losses_124789l;?8
1?.
(?%
inputs?????????@
p
? "-?*
#? 
0?????????@
? ?
+__inference_dropout_10_layer_call_fn_124794_;?8
1?.
(?%
inputs?????????@
p 
? " ??????????@?
+__inference_dropout_10_layer_call_fn_124799_;?8
1?.
(?%
inputs?????????@
p
? " ??????????@?
F__inference_dropout_11_layer_call_and_return_conditional_losses_124835^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
F__inference_dropout_11_layer_call_and_return_conditional_losses_124847^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
+__inference_dropout_11_layer_call_fn_124852Q4?1
*?'
!?
inputs??????????
p 
? "????????????
+__inference_dropout_11_layer_call_fn_124857Q4?1
*?'
!?
inputs??????????
p
? "????????????
E__inference_dropout_9_layer_call_and_return_conditional_losses_124690l;?8
1?.
(?%
inputs????????? 
p 
? "-?*
#? 
0????????? 
? ?
E__inference_dropout_9_layer_call_and_return_conditional_losses_124702l;?8
1?.
(?%
inputs????????? 
p
? "-?*
#? 
0????????? 
? ?
*__inference_dropout_9_layer_call_fn_124707_;?8
1?.
(?%
inputs????????? 
p 
? " ?????????? ?
*__inference_dropout_9_layer_call_fn_124712_;?8
1?.
(?%
inputs????????? 
p
? " ?????????? ?
E__inference_flatten_3_layer_call_and_return_conditional_losses_124805a7?4
-?*
(?%
inputs?????????@
? "&?#
?
0?????????? 
? ?
*__inference_flatten_3_layer_call_fn_124810T7?4
-?*
(?%
inputs?????????@
? "??????????? ?
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_124670?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_124675h7?4
-?*
(?%
inputs?????????   
? "-?*
#? 
0????????? 
? ?
0__inference_max_pooling2d_6_layer_call_fn_124680?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
0__inference_max_pooling2d_6_layer_call_fn_124685[7?4
-?*
(?%
inputs?????????   
? " ?????????? ?
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_124757?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_124762h7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????@
? ?
0__inference_max_pooling2d_7_layer_call_fn_124767?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
0__inference_max_pooling2d_7_layer_call_fn_124772[7?4
-?*
(?%
inputs?????????@
? " ??????????@?
D__inference_model_45_layer_call_and_return_conditional_losses_124189u !"#$>?;
4?1
'?$
input?????????  
p 

 
? "%?"
?
0?????????

? ?
D__inference_model_45_layer_call_and_return_conditional_losses_124218u !"#$>?;
4?1
'?$
input?????????  
p

 
? "%?"
?
0?????????

? ?
D__inference_model_45_layer_call_and_return_conditional_losses_124302v !"#$??<
5?2
(?%
inputs?????????  
p 

 
? "%?"
?
0?????????

? ?
D__inference_model_45_layer_call_and_return_conditional_losses_124376v !"#$??<
5?2
(?%
inputs?????????  
p

 
? "%?"
?
0?????????

? ?
)__inference_model_45_layer_call_fn_124044h !"#$>?;
4?1
'?$
input?????????  
p 

 
? "??????????
?
)__inference_model_45_layer_call_fn_124160h !"#$>?;
4?1
'?$
input?????????  
p

 
? "??????????
?
)__inference_model_45_layer_call_fn_124405i !"#$??<
5?2
(?%
inputs?????????  
p 

 
? "??????????
?
)__inference_model_45_layer_call_fn_124434i !"#$??<
5?2
(?%
inputs?????????  
p

 
? "??????????
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_123944 !"#$H?E
>?;
1?.
conv2d_12_input?????????  
p 

 
? "%?"
?
0?????????

? ?
H__inference_sequential_3_layer_call_and_return_conditional_losses_123984 !"#$H?E
>?;
1?.
conv2d_12_input?????????  
p

 
? "%?"
?
0?????????

? ?
H__inference_sequential_3_layer_call_and_return_conditional_losses_124493v !"#$??<
5?2
(?%
inputs?????????  
p 

 
? "%?"
?
0?????????

? ?
H__inference_sequential_3_layer_call_and_return_conditional_losses_124567v !"#$??<
5?2
(?%
inputs?????????  
p

 
? "%?"
?
0?????????

? ?
-__inference_sequential_3_layer_call_fn_123632r !"#$H?E
>?;
1?.
conv2d_12_input?????????  
p 

 
? "??????????
?
-__inference_sequential_3_layer_call_fn_123904r !"#$H?E
>?;
1?.
conv2d_12_input?????????  
p

 
? "??????????
?
-__inference_sequential_3_layer_call_fn_124596i !"#$??<
5?2
(?%
inputs?????????  
p 

 
? "??????????
?
-__inference_sequential_3_layer_call_fn_124625i !"#$??<
5?2
(?%
inputs?????????  
p

 
? "??????????
?
$__inference_signature_wrapper_124249? !"#$??<
? 
5?2
0
input'?$
input?????????  ";?8
6
sequential_3&?#
sequential_3?????????
