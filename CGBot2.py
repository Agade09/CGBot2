#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import re as regex
from slixmpp import ClientXMPP
import datetime
import os
import functools
from collections import Counter

seq_length = 1000 # The maximum length sentence we want for a single input in characters
BATCH_SIZE = 2
BUFFER_SIZE = 10000
EPOCHS=100
embedding_dim = 256 # The embedding dimension 
rnn_units = 1024 # Number of RNN units
Training_Proportion = 0.95
load_checkpoint = True
Train = False
Log_Messages=True
Test=False
Initialisation_Length=1000
Vocab_Limit=256
Temperature = 0.25 # Low temperatures results in more predictable text. Higher temperatures results in more surprising text. Experiment to find the best setting.
checkpoint_dir = './training_checkpoints' # Directory where the checkpoints will be saved
logs_dir = './Logs'
config_filename = 'Config.txt'

if not Train:
  tf.enable_eager_execution()

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

def build_model(vocab_size, embedding_dim, batch_size):
  if Train:
    rnn = tf.keras.layers.CuDNNGRU
  else:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    rnn = functools.partial(tf.keras.layers.GRU,reset_after=True,recurrent_activation='sigmoid')
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size,embedding_dim,batch_input_shape=[batch_size,None]),
    rnn(rnn_units,return_sequences=True,recurrent_initializer='glorot_uniform',stateful=True),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

def loss(labels, logits):
  return tf.keras.backend.sparse_categorical_crossentropy(labels,logits,from_logits=True)

def Predictions_To_Id(predictions):
  predictions = tf.squeeze(predictions, 0) # remove the batch dimension
  # using a multinomial distribution to predict the word returned by the model
  predictions = predictions / Temperature
  predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()
  return predicted_id

def Predictions_To_Char(predictions,idx2char):
  return idx2char[Predictions_To_Id(predictions)]

def String_To_Int_Vector(string,char2idx):
  return np.array([(char2idx[c] if (c in char2idx) else len(char2idx)-1) for c in string],dtype=np.uint8)

def Feed_Model(model,message,char2idx):
  msg_input = String_To_Int_Vector(message,char2idx)
  msg_input = tf.expand_dims(msg_input, 0)
  predictions = model(msg_input)
  return predictions

def generate_response(model,start_string,idx2char,char2idx): # Evaluation step (generating text using the learned model)
  start = time.time()
  num_generate = 100 # Max Number of characters to generate
  input_eval = String_To_Int_Vector(start_string,char2idx)
  input_eval = tf.expand_dims(input_eval, 0)
  text_generated = []
  for i in range(num_generate):
      predictions = model(input_eval)
      predicted_id = Predictions_To_Id(predictions)
      
      # We pass the predicted word as the next input to the model along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)
      char_generated = idx2char[predicted_id]
      if char_generated=='\n': #Not working for some reason
        Feed_Model(model,'\n',char2idx)
        print("Took "+str(time.time()-start)+" to generate a response")
        break
      text_generated.append(char_generated)
  return ''.join(text_generated)

def Filter_Logs(logs):
  timestamp = regex.compile(r'\(\d\d:\d\d:\d\d\)')
  All_lines=logs.split("\n")
  Filtered_Logs=""
  for line in All_lines:
    if len(line)>0:
      match = timestamp.search(line)
      if match!=None and match.span()[0]==0:#Contains timestamp?
        first_parenthesis=line.find(')')
        username_colon=line.find(':',first_parenthesis)
        Filtered_Logs+='\n'+line[first_parenthesis+2:username_colon-1]+':'+line[username_colon+2:]
      else:
        Filtered_Logs+=' '+line
  return Filtered_Logs

def Log_Message(msg):
  if Log_Messages:
    log_filename=msg['from'].bare+'-'+datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d')+'.log'
    log_file=open(logs_dir+'/'+log_filename,'a+')
    regex.sub(r"\r\n"," ",msg['body']) # Get rid of newlines and carriage returns
    if msg.get_mucnick()!='':
      log_file.write(datetime.datetime.fromtimestamp(time.time()).strftime('(%H:%M:%S)')+' '+msg.get_mucnick()+' : '+msg['body']+'\n')
    else:
      print("msg.get_mucnick() returned ''")
      print(msg)

class ChannelBot(ClientXMPP):
    def __init__(self, jid, password,nick,rooms,MUC_name):
      ClientXMPP.__init__(self, jid, password)
      self.nickname=nick
      self.room_names=rooms
      self.MUC=MUC_name

      self.model = {}
      self.char2idx={}
      self.idx2char={}
      for room_name in self.room_names:
        vocab_filepath = checkpoint_dir+'/vocab_'+room_name+'.npy'
        if os.path.isfile(vocab_filepath):
          vocab = np.load(vocab_filepath) # The unique characters in the file
        else:
          print("Could not find vocab file at: "+vocab_filepath)
          exit()
        vocab_size = len(vocab) # Length of the vocabulary in chars
        # Creating a mapping from unique characters to indices
        self.char2idx[room_name] = {u:i for i, u in enumerate(vocab)}
        self.idx2char[room_name] = np.array(vocab)

        self.model[room_name] = build_model(vocab_size,embedding_dim,batch_size=1)
        checkpoint_file = checkpoint_dir+"/weights_"+room_name+".h5"
        if os.path.isfile(checkpoint_file):
          self.model[room_name].load_weights(checkpoint_file)
        else:
          print("Could not find checkpoint file: "+checkpoint_file)
          exit()
        self.model[room_name].build(tf.TensorShape([1, None]))
        text=""
        logfile = regex.compile(regex.escape(room_name)+r'@'+regex.escape(MUC)+r'-\d\d\d\d-\d\d-\d\d\.log')
        for file in sorted(os.listdir(logs_dir),reverse=True):
          match = logfile.search(file)
          if match!=None and match.span()[0]==0:
            print("Parsing "+file+" for inference initialisation")
            text=open(logs_dir+'/'+file).read()+text
            if len(text)>=Initialisation_Length:
              break
        text = text[len(text)-Initialisation_Length:]
        Feed_Model(self.model[room_name],text,self.char2idx[room_name])#Give the model some state
      if Test:
        first_room=self.room_names[0]
        Test_Bot(self.model[first_room],self.idx2char[first_room],self.char2idx[first_room])
      self.register_plugin('xep_0045')
      self.add_event_handler("session_start", self.session_start)
      self.add_event_handler("message", self.message)
      self.add_event_handler("groupchat_message", self.muc_message)

    def session_start(self, event):
      self.send_presence()
      self.get_roster()
      for room_name in self.room_names:
        self.plugin['xep_0045'].join_muc(room_name+'@'+self.MUC,self.nickname,wait=True)
      print("Session started")

    def message(self, msg): #Private message?
      if msg['type'] in ('chat', 'normal'):
        first_room=self.room_names[0]
        reply=self.make_message(mto=msg['from'].bare,mbody=generate_response(self.model[first_room],self.nickname+':',self.idx2char[first_room],self.char2idx[first_room]))
        reply['id']=first_room+"_"+self.nickname+"@"+self.MUC+"/"+datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        reply.send()

    def muc_message(self,msg):
      room_name=msg['from'].bare
      room_name=room_name[:room_name.find('@')]
      print(room_name)
      print(msg.get_mucnick()+':'+msg['body'])
      Log_Message(msg)
      Feed_Model(self.model[room_name],msg.get_mucnick()+':'+msg['body']+'\n',self.char2idx[room_name])
      if msg.get_mucnick()!=self.nickname and self.nickname.lower() in msg['body'].lower():
        print("Saw my nickname")
        reply=self.make_message(mto=msg['from'].bare,mbody=generate_response(self.model[room_name],self.nickname+':',self.idx2char[room_name],self.char2idx[room_name]),mtype='groupchat')
        reply['id']=room_name+"_"+self.nickname+"@"+self.MUC+"/"+datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        reply.send()

def Train_Bot(channel_name,MUC):
  text=""
  logfile = regex.compile(regex.escape(channel_name)+r'@'+regex.escape(MUC)+r'-\d\d\d\d-\d\d-\d\d\.log')
  for file in sorted(os.listdir(logs_dir)):
    match = logfile.search(file)
    if match!=None and match.span()[0]==0:
      text+=open(logs_dir+'/'+file).read()
  if len(text)<=0:
    print("Didn't find any logs for channel: "+channel_name+"@"+MUC)
    exit()
  text=Filter_Logs(text)
  #print(text)
  #vocab = np.array(sorted(set(text))) # The unique characters in the file
  vocab = Counter(text)
  vocab = sorted(vocab, key=vocab.get, reverse=True)
  vocab = np.array(vocab[:Vocab_Limit])
  #print(Counter(text))
  #print(vocab)
  np.save(checkpoint_dir+'/vocab_'+channel_name,vocab)
  vocab_size = len(vocab) # Length of the vocabulary in chars
  # Creating a mapping from unique characters to indices
  char2idx = {u:i for i, u in enumerate(vocab)}
  idx2char = np.array(vocab)

  text=text.split('\n')
  def Pad(string):
    if len(string)>seq_length:
      return string[:seq_length]
    else:
      return string.ljust(seq_length) #pad with spaces
  def Sample(text_by_line):
    text_samples=[]
    i=0
    while i<len(text_by_line):
      #print(str(i)+"\\"+str(len(text_by_line)))
      sample=""
      while i<len(text_by_line) and len(sample)+len(text_by_line[i])+1<=seq_length: #How to deal with a really long line?
        sample+=text_by_line[i]+'\n'
        i+=1
      text_samples.append(Pad(sample))
    return text_samples
  text = Sample(text)
  text_as_int = np.array([String_To_Int_Vector(s,char2idx) for s in text])
  #text_as_int = [[text_as_int[i],text_as_int[i+1]] for i in range(len(text_as_int)-1)]
  #print(text_as_int)
  training_cutoff=round(len(text_as_int)*Training_Proportion)
  train_text = text_as_int[:training_cutoff]
  #print(train_text)
  validation_text = text_as_int[training_cutoff:]

  # Create training examples / targets
  dataset = tf.data.Dataset.from_tensor_slices(train_text)
  #dataset = dataset.batch(seq_length+1,drop_remainder=True)
  dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(BUFFER_SIZE))
  dataset = dataset.apply(tf.data.experimental.map_and_batch(batch_size=BATCH_SIZE,drop_remainder=True,map_func=split_input_target))
  #dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

  validation_dataset = tf.data.Dataset.from_tensor_slices(validation_text)
  #validation_dataset = validation_dataset.batch(seq_length+1,drop_remainder=True)
  validation_dataset = validation_dataset.apply(tf.data.experimental.map_and_batch(batch_size=BATCH_SIZE,drop_remainder=True,map_func=split_input_target))
  validation_dataset = validation_dataset.repeat()
  #validation_dataset = validation_dataset.prefetch(tf.contrib.data.AUTOTUNE)

  model = build_model(vocab_size=len(vocab),embedding_dim=embedding_dim,batch_size=BATCH_SIZE)
  checkpoint_file = checkpoint_dir+"/weights_"+channel_name+".h5"
  if load_checkpoint and os.path.isfile(checkpoint_file):
    model.load_weights(checkpoint_file)
    print("Restarting training from previous checkpoint")
  else:
    print("Training from random weights")
  model.summary()
  model.compile(optimizer=tf.train.AdamOptimizer(),loss=loss)
  checkpoint_prefix = checkpoint_dir+"/weights_"+channel_name+".h5"# Name of the checkpoint files
  checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,save_weights_only=True,save_best_only=True,monitor='val_loss')

  steps_per_training_epoch = len(train_text)//BATCH_SIZE
  steps_per_validation_epoch = len(validation_text)//BATCH_SIZE
  training_history = model.fit(dataset, epochs=EPOCHS, steps_per_epoch=steps_per_training_epoch, callbacks=[checkpoint_callback], validation_steps=steps_per_validation_epoch,validation_data=validation_dataset)

  # summarize history for loss
  plt.plot(training_history.history['loss'])
  plt.plot(training_history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.savefig('loss.png')
  plt.gcf().clear()

def Test_Bot(model,idx2char,char2idx):
  Phrases = ["Agade:Salut tout le monde","Agade:J'ai un probleme sur temperatures"]
  for phrase in Phrases:
    for i in range(10):
      Feed_Model(model,phrase,char2idx)
      response = generate_response(model,"NeumaNN: ",idx2char,char2idx)
      print("Phrase: "+phrase)
      print(response)

config_file = open(config_filename,"r")
CG_ID = config_file.readline().split()[0]
CG_password = config_file.readline().split()[0]
Chat_host = config_file.readline().split()[0]
Chat_port= config_file.readline().split()[0]
MUC = config_file.readline().split()[0]
Nickname = config_file.readline().split()[0]
Channel_line=config_file.readline()
Channels = Channel_line[:Channel_line.find("//")].split()
config_file.close()

if Train:
  Train_Bot(Channels[0],MUC)
else:
  Bot = ChannelBot(CG_ID+'@'+Chat_host,CG_password,Nickname,Channels,MUC)
  Bot.connect()
  Bot.process()