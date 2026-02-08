# Task 4: Machine Translation Web Application

A simple web application was developed to demonstrate the trained
Filipino–English neural machine translation model. The frontend was built
using HTML, CSS, and JavaScript, while the backend was implemented using
Python Flask.

The application accepts a Filipino sentence as input and generates an
English translation using an LSTM-based encoder–decoder model with
additive attention. This attention mechanism was selected based on its
superior performance in Task 2 and Task 3.

The frontend communicates with the backend via HTTP POST requests, and
the translated output is displayed in real time.
