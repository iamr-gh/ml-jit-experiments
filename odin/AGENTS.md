# Odin ML Compiler Experiments

## Overview
This is a repository is a collection of experiments around creating an end-end ML jit compiler using Odin. 
This will eventually include, but is not limited to, a gradient engine, a tensor/layout library, an optimizer, and a high level interface to define neural networks and similar operations.

## Specific Rules
Do not generate extraneous comments in the codebase. 
Strive to explain code with descriptive names and simple logic.
Only use comments when meaning would not be clear to a human who was an expert.

## Files

``grad_single.odin`` specified a gradient engine for single variable expressions, where eval_grad_forward_all and eval_grad_reverse preform automatic forward and backward differentiation.

``matrix_of_single.odin`` contains an implementation of matrix expressions written as a combination of single variable expressions.

``test.odin`` contains a framework for timing the execution of runs, and has various runtime tests for evaluating the speed of different vm_generated code.
