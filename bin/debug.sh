#!/bin/sh
sudo make clean
make BUILD=debug -j5
sudo make BUILD=debug -j5 install
sudo ldconfig
make BUILD=debug -j5 example
chown -R cory:cory build/*
ulimit -c unlimited

