#!/bin/python3
import os
from subprocess import *


def run_cmd_small_output(cmd):
	print(f"Running command: {cmd}\n")
	Popen(cmd, shell=True, stdout=PIPE).communicate()

def run_cmd_to_file(cmd, sf_out):
	print(f"Running command with output:\n{cmd}\nto {sf_out}\n")
	sf_err=sf_out+".err"
	errcode=None
	with open(sf_out, "w") as f:
		p = Popen(cmd, shell=True, stdout=f, stderr=None)
		p.communicate()
	if os.path.isfile(sf_err):
		os.remove(sf_err)
	return errcode

def run_cmd_small_output_with_blocking(cmd):
	print(f"Running command: {cmd}\n")
	call(cmd, shell=True, stdout=None)

def run_cmd_to_file_with_blocking(cmd, sf_out):
	print(f"Running command with output:\n{cmd} > {sf_out}\nWait until this process is finished.")
	sf_err=sf_out+".err"
	errcode=None
	with open(sf_out, "w") as f:
		p = call(cmd, shell=True, stdout=f, stderr=None)
	if os.path.isfile(sf_err):
		os.remove(sf_err)
	return errcode
