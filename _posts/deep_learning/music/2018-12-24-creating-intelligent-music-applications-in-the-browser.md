---
layout: post
category: software
class: Music
title: Creating Intelligent Music Applications in the Browser
description: Let's learn and understand Google's TensorFlow.js and Magenta.js to create AI music applications in the browser.
author: Gogul Ilango
permalink: software/creating-intelligent-music-applications-in-the-browser
image: https://drive.google.com/uc?id=1f4GHFHGhE7htiJ4adyFr2wYScjfvIU1A
cardimage: https://drive.google.com/uc?id=1f4GHFHGhE7htiJ4adyFr2wYScjfvIU1A
---

<div class="git-showcase">
  <div>
    <a class="github-button" href="https://github.com/Gogul09" data-show-count="true" aria-label="Follow @Gogul09 on GitHub">Follow @Gogul09</a>
  </div>

  <div>
	<a class="github-button" href="https://github.com/Gogul09/deep-drum/fork" data-icon="octicon-repo-forked" data-show-count="true" aria-label="Fork Gogul09/deep-drum on GitHub">Fork</a>
  </div>

  <div>
	<a class="github-button" href="https://github.com/Gogul09/deep-drum" data-icon="octicon-star" data-show-count="true" aria-label="Star Gogul09/deep-drum on GitHub">Star</a>
  </div>  
</div>

<div class="sidebar_tracker" id="sidebar_tracker">
  <button onclick="closeSidebar('sidebar_tracker_content')">X</button>
  <p onclick="showSidebar('sidebar_tracker_content')">Contents</p>
  <ul id="sidebar_tracker_content">
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_1" href="#why-music-and-ml">Why Music and ML?</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_2" href="#why-browser-for-ml">Why browser for ML?</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_3" href="#using-magentas-pre-trained-models">Using Magenta's Pre-trained Models</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_4" href="#generating-drum-patterns-using-drumsrnn">Generating Drum Patterns using DrumsRNN</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_5" href="#cool-demos">Cool Demos</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_6" href="#resources">Resources</a></li>
  </ul>
</div>

After the introduction of Google's [TensorFlow.js](https://js.tensorflow.org/){:target="_blank"}, it has become a lot easier to make use of browser (client-side) to do Deep learning. There are handy approaches (as discussed [here](https://js.tensorflow.org/tutorials/import-keras.html){:target="_blank"}) on deploying deep learning models using [Keras](https://keras.io/){:target="_blank"} and TensorFlow.js.

<div class="note">
	<p><b>Note</b>: To learn more about TensorFlow.js and its applications, kindly visit this <a href="https://github.com/tensorflow/tfjs/blob/master/GALLERY.md" target="_blank">link</a>.</p>
</div>


<h3 id="why-music-and-ml">Why Music and ML?</h3>


Music generation has already began to catch the eyes of machine learning devs and there are numerous projects that are getting pushed in [GitHub](https://github.com/search?q=music+generation){:target="_blank"} every week. Although there exist a barrier between AI Researchers and AI Developers such as complex mathematics that involve derivations and jargons, there is still hope for an AI enthusiast to use code and some music knowledge to create exciting applications that was a dream few years back. 

Leveraging the capabilities of TensorFlow.js, we now have Google's [Magenta.js](https://magenta.tensorflow.org/){:target="_blank"} using which any developer with knowledge on javascript and music could create a music application that has intelligence built in to it.

I loved the concept behind [Google's Magenta team](https://ai.google/research/teams/brain/magenta/){:target="_blank"}.

> When a painter creates a work of art, she first blends and explores color options on an artist’s palette before applying them to the canvas. This process is a creative act in its own right and has a profound effect on the final work. Musicians and composers have mostly lacked a similar device for exploring and mixing musical ideas, but we are hoping to change that - [Read more](https://magenta.tensorflow.org/music-vae){:target="_blank"}

<h3 id="why-browser-for-ml">Why browser for ML?</h3>

Although one might feel that browsers are light-weight apps that wont handle data intensive algorithms such as deep neural networks, by leveraging the capabilities of [WebGL](https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API){:target="_blank"} in browsers such as Google Chrome, we could create, train and deploy deep neural net models right in the browser itself without any server requests.

Another advantage of using browser to create AI applications is that you could easily send your creations to your friends or family using nothing more than a simple link!

<h3 id="using-magentas-pre-trained-models">Using Magenta's Pre-trained Models</h3>

Magenta.js is a javascript library that is built on top of TensorFlow.js which provides music oriented abstraction for developers. Google's Magenta Team research, create and train deep learning models such as Long-Short Term Memory nets (LSTMs), Variational Auto-Encoders (VAE) etc., for music generation, and serve those models as pre-trained models for an AI enthusiast like me to use it for free.

By using a pre-trained magenta model, we could build creative music applications in the browser using deep learning. Some of the note-based music models that are provided by Magenta are MusicVAE, MelodyRNN, DrumsRNN and ImprovRNN. Using these pretrained models, we could use their [magenta.js](https://github.com/tensorflow/magenta-js/){:target="_blank"} API to create cool music apps.

<div class="note">
<p><b>Note</b>: The prerequisites for making an application using Magenta.js include knowledge on HTML, CSS and JavaScript.</p>
</div>

<h3 id="generating-drum-patterns-using-drumsrnn">Generating Drum Patterns using DrumsRNN</h3>

In this tutorial, I will show you how to create an intelligent music application that I call [DeepDrum & DeepArp](https://gogul09.github.io/software/deep-drum){:target="_blank"} using javascript and Google's magenta.js in the browser. First, we will focus on generating drum patterns using magenta's [drums_rnn](https://github.com/tensorflow/magenta/tree/master/magenta/models/drums_rnn){:target="_blank"}. Similar approach is used to create arpeggio patterns using magenta's [improv_rnn](https://github.com/tensorflow/magenta/tree/master/magenta/models/improv_rnn){:target="_blank"}.

<figure>
	<img src="https://drive.google.com/uc?id=1f4GHFHGhE7htiJ4adyFr2wYScjfvIU1A" />
	<figcaption>DeepDrum & DeepArp using Google Magenta's DrumsRnn & ImprovRNN</figcaption>
</figure>

The core algorithm behind these two models is the special type Recurrent Neural Network named Long-Short Term Memory network (LSTM). You can read more about the inner workings of an LSTM network in this excellent [blog post](http://colah.github.io/posts/2015-08-Understanding-LSTMs/){:target="_blank"} by Christopher Olah.

---

**Step 1**: To include Magenta.js for your music application, you simply need to include the following script in your html **head** tag.

<div class="code-head"><span>code</span>index.html</div>

```html
<script type="text/javascript" src="https://cdn.jsdelivr.net/npm/@magenta/music@1.4.2/dist/magentamusic.min.js"></script>
```

**Step 2**: A pretrained magenta model can easily be loaded into a javascript environment using the [js-checkpoints](https://github.com/tensorflow/magenta-js/blob/master/music/checkpoints/README.md){:target="_blank"} (that magenta team has made publicly available) which automatically loads the model along with config files in a single line of code.

<div class="code-head"><span>code</span>app.js</div>

```javascript
let drums_rnn = mm.MusicRNN("https://storage.googleapis.com/download.magenta.tensorflow.org/tfjs_checkpoints/music_rnn/drum_kit_rnn");
```

There is a tradeoff between model package size and accuracy as inference of a pre-trained model is happening live in the browser (client-side).

**Step 3**: Next, we need to initialize the model to make use of its methods and attributes.

<div class="code-head"><span>code</span>app.js</div>

```javascript
drums_rnn.initialize();
```

**Step 4**: Our drum pattern generator will work like this - You provide some random input seed pattern first and the deep neural network (DrumsRNN) will predict the next sequence of patterns.

We have 10 individual drumkit instrument such as <span class="coding">kick</span>, <span class="coding">snare</span>, <span class="coding">hihat closed</span>, <span class="coding">hihat open</span>, <span class="coding">tom low</span>, <span class="coding">tom mid</span>, <span class="coding">tom high</span>, <span class="coding">clap</span> and <span class="coding">ride</span>. Hence, we define an array named <span class="coding">seed_pattern</span> to hold the **ON** time step of each instrument (in an array) at every time step.

For example, I have initialized the <span class="coding">seed_pattern</span> as shown below. This means, for a <span class="coding">seed_limit</span> of 4 time steps, we assign the input pattern like this - 

* <span class="coding">kick</span> should be **ON** at first and third time step.
* <span class="coding">snare</span> shouldn't be turned **ON** within <span class="coding">seed_limit</span>.
* <span class="coding">hihat closed</span> should be **ON** only at third time step.
* and so on..

Notice we start first time step as 0 in code.

<div class="code-head"><span>code</span>app.js</div>

```javascript
var seed_pattern = [
	[0, 2], 
	[], 
	[2], 
	[], 
	[2], 
	[], 
	[0, 2], 
	[], 
	[1, 2], 
	[]
];
```

With the input <span class="coding">seed_pattern</span> and <span class="coding">seed_limit</span> defined, we could simply ask our <span class="coding">drums_rnn</span> to continue the sequence for us. Before doing that, we need to be aware of [quantization](https://en.wikipedia.org/wiki/Quantization){:target="_blank"} of the input values.

**Step 5**: To quantize the note sequence, we feed the input <span class="coding">seed_pattern</span> into a javascript object as shown below.

<div class="code-head"><span>code</span>app.js</div>

```javascript
let cur_seq = drum_to_note_sequence(seed_pattern);

//---------------------------------
// drum to note sequence formation
//---------------------------------
function drum_to_note_sequence(quantize_tensor) {
	var notes_array = [];
	var note_index = 0;
	for (var i = 0; i < quantize_tensor.length; i++) {
		var notes = quantize_tensor[i];
		if(notes.length > 0) {
			for (var j = 0; j < notes.length; j++) {
				notes_array[note_index] = {};
				notes_array[note_index]["pitch"] = midiDrums[notes[j]];
				notes_array[note_index]["startTime"] = i * 0.5;
				notes_array[note_index]["endTime"] = (i+1) * 0.5;
				note_index = note_index + 1;
			}
		}
	}

	return mm.sequences.quantizeNoteSequence(
	  {
	    ticksPerQuarter: 220,
	    totalTime: quantize_tensor.length / 2,
	    timeSignatures: [
	      {
	        time: 0,
	        numerator: ts_num,
	        denominator: ts_den
	      }
	    ],
	    tempos: [
	      {
	        time: 0,
	        qpm: tempo
	      }
	    ],
	    notes: notes_array
	   },
	  1
	);
}
```

The way I figured out what's inside <span class="coding">mm.sequences.quantizeNoteSequence</span> is through the browser's console and some help by looking at the code of few [demos](https://magenta.tensorflow.org/demos){:target="_blank"} in Magenta's website. Values like <span class="coding">timeSignatures</span>, <span class="coding">tempos</span> and <span class="coding">totalTime</span> need to be set according to one's preferences. You could even assign these values dynamically.

The main thing you need to take care here is the conversion of our input <span class="coding">seed_pattern</span> into musical quantization format that Magenta accepts which includes defining each drumkit instrument's <span class="coding">pitch</span>, <span class="coding">startTime</span> and <span class="coding">endTime</span>. 

<span class="coding">pitch</span> of a drumkit note is the MIDI value of that note which could be obtained from this [mapping](https://github.com/Gogul09/deep-drum/blob/master/js/reverse_midi_map.js){:target="_blank"}. <span class="coding">startTime</span> and <span class="coding">endTime</span> are quantization values that defines the start and end time for a single note. 

For example, for our first time step, <span class="coding">kick</span> will have the following values.
* <span class="coding">pitch</span> - 36
* <span class="coding">startTime</span> - 0
* <span class="coding">endTime</span> - 0.5

**Step 6**: Once you have encoded the input <span class="coding">seed_pattern</span> to Magenta's quantization format, you can ask <span class="coding">drums_rnn</span> to continue the sequence as shown below.

<div class="code-head"><span>code</span>app.js</div>

```javascript
const player_length = 32;
const temperature_drum = 1;

predicted_sequence = await drums_rnn
		.continueSequence(cur_seq, player_length, temperature_drum)
		.then(r => seed_pattern.concat(note_to_drum_sequence(r, player_length)));

//---------------------------------
// note to drum sequence formation
//---------------------------------
function note_to_drum_sequence(seq, pLength) {
	let res = [];
	for (var i = 0; i < pLength; i++) {
		empty_list = [];
		res.push(empty_list);
	}
    for (let { pitch, quantizedStartStep } of seq.notes) {
      res[quantizedStartStep].push(reverseMidiMapping.get(pitch));
    }
    return res;
}
```

First, we use <span class="coding">continueSequence()</span> function of <span class="coding">drums_rnn</span> to predict the next sequence values for all our drumkit instruments and store it in a variable named <span class="coding">predicted_sequence</span>. These predictions will be based on the same old magenta's quantization format having MIDI-mapped pitch values, start time and end time. 

We define an array named <span class="coding">res</span> and store the predicted sequence values based on its <span class="coding">quantizedStartStep</span>. We then concatenate the predicted sequence with the input <span class="coding">seed_pattern</span> to generate a beat!

---

These are the core steps involved in using Google Magenta's pretrained model to generate sequences for music generation. You can use the same steps to generate arpeggio patterns using [improv_rnn](https://github.com/tensorflow/magenta/tree/master/magenta/models/improv_rnn){:target="_blank"}.

You can check the entire code that I have used to build this music application [here](https://github.com/Gogul09/deep-drum){:target="_blank"}.

<div class="note">
<p><b>Note</b>: If you still don't understand the steps mentioned here, I highly encourage you to do <span class="coding">console.log()</span> at each step of the code and understand the steps completely.</p>
</div>

<h3 id="cool-demos">Cool Demos</h3>

People working in this domain are musicians, artists, creative coders, programmers and researchers who have built extremely amazing demos that you can find [here](https://magenta.tensorflow.org/demos){:target="_blank"} and [here](https://aijs.rocks/){:target="_blank"}.

<h3 id="resources">Resources</h3>

* [Magenta.js](https://magenta.tensorflow.org/js-announce){:target="_blank"}
* [TensorFlow.js Gallery](https://github.com/tensorflow/tfjs/blob/master/GALLERY.md){:target="_blank"}
* [Music and AI in the Browser with TensorFlow js and Magenta js – Tero Parviainen](https://www.youtube.com/watch?v=GJfjKdpmN6g){:target="_blank"}
* [Musical Deep Neural Networks in the Browser by Tero Parviainen](https://www.youtube.com/watch?v=HKRJuz6o2uY){:target="_blank"}
* [AI JavaScript Rocks - Asim Hussain](https://www.youtube.com/watch?v=FgoVL3A6RCo){:target="_blank"}