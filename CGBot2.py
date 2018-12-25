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

tf.enable_eager_execution()

seq_length = 1000 # The maximum length sentence we want for a single input in characters
BATCH_SIZE = 64
BUFFER_SIZE = 10000
EPOCHS=100
embedding_dim = 256 # The embedding dimension 
rnn_units = 1024 # Number of RNN units
Training_Proportion = 0.95
load_checkpoint = True
Train = False
Log_Messages=True
Initialisation_Length=1000
Temperature = 0.25 # Low temperatures results in more predictable text. Higher temperatures results in more surprising text. Experiment to find the best setting.
checkpoint_dir = './training_checkpoints' # Directory where the checkpoints will be saved
logs_dir = './Logs'
config_filename = 'Config.txt'

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
  return [(char2idx[c] if (c in char2idx) else len(char2idx)-1) for c in string]

def Feed_Model(model,message,char2idx):
  msg_input = String_To_Int_Vector(message,char2idx)
  msg_input = tf.expand_dims(msg_input, 0)
  predictions = model(msg_input)
  return predictions

def generate_response(model,start_string,idx2char,char2idx): # Evaluation step (generating text using the learned model)
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
        break
      text_generated.append(char_generated)
  return ''.join(text_generated)

def Filter_Logs(logs): #Logs didn't seem to contain '\r'
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
    print(log_filename)
    log_file=open(logs_dir+'/'+log_filename,'a+')
    regex.sub(r"\r\n"," ",msg['body']) # Get rid of newlines and carriage returns
    if msg.get_mucnick()!='':
      log_file.write(datetime.datetime.fromtimestamp(time.time()).strftime('(%H:%M:%S)')+' '+msg.get_mucnick()+' : '+msg['body'])
    else:
      print("msg.get_mucnick() returned ''")
      print(msg)

class ChannelBot(ClientXMPP):
    def __init__(self, jid, password,nick,room,MUC_name):
      ClientXMPP.__init__(self, jid, password)
      self.nickname=nick
      self.room_name=room
      self.MUC=MUC_name

      vocab_filepath = checkpoint_dir+'/vocab_'+self.room_name+'.npy'
      if os.path.isfile(vocab_filepath):
        vocab = np.load(vocab_filepath) # The unique characters in the file
      else:
        print("Could not find vocab file at: "+vocab_filepath)
        exit()
      self.vocab_size = len(vocab) # Length of the vocabulary in chars
      # Creating a mapping from unique characters to indices
      self.char2idx = {u:i for i, u in enumerate(vocab)}
      self.idx2char = np.array(vocab)

      self.model = build_model(self.vocab_size,embedding_dim,batch_size=1)
      checkpoint_file = checkpoint_dir+"/weights_"+self.room_name+".h5"
      if os.path.isfile(checkpoint_file):
        self.model.load_weights(checkpoint_file)
      else:
        print("Could not find checkpoint file: "+checkpoint_file)
        exit()
      self.model.build(tf.TensorShape([1, None]))
      text=""
      logfile = regex.compile(regex.escape(self.room_name)+r'@'+regex.escape(MUC)+r'-\d\d\d\d-\d\d-\d\d\.log')
      for file in sorted(os.listdir(logs_dir),reverse=True):
        match = logfile.search(file)
        if match!=None and match.span()[0]==0:
          print("Parsing "+file+" for inference initialisation")
          text=open(logs_dir+'/'+file).read()+text
          if len(text)>=Initialisation_Length:
            break
      text = text[len(text)-Initialisation_Length:]
      Feed_Model(self.model,text,self.char2idx)#Give the model some state

      self.register_plugin('xep_0045')
      self.add_event_handler("session_start", self.session_start)
      self.add_event_handler("message", self.message)
      self.add_event_handler("groupchat_message", self.muc_message)

    def session_start(self, event):
      self.send_presence()
      self.get_roster()
      self.plugin['xep_0045'].join_muc(self.room_name+'@'+self.MUC,self.nickname,wait=True)
      print("Session started")

    def message(self, msg): #Private message?
      if msg['type'] in ('chat', 'normal'):
          msg.reply("Thanks for sending\n%(body)s" % msg).send()

    def muc_message(self,msg):
      print(msg.get_mucnick()+':'+msg['body'])
      Log_Message(msg)
      Feed_Model(self.model,msg.get_mucnick()+':'+msg['body'],self.char2idx)
      if msg.get_mucnick()!=self.nickname and self.nickname.lower() in msg['body'].lower():
        print("Saw my nickname")
        #print(self.room_name+"@"+self.MUC+"/"+datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
        reply=self.make_message(mto=msg['from'].bare,mbody=generate_response(self.model,'\n'+self.nickname+':',self.idx2char,self.char2idx),mtype='groupchat')
        reply['id']=self.room_name+"_"+self.nickname+"@"+self.MUC+"/"+datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        reply.send()
      else:
        Feed_Model(self.model,'\n',self.char2idx)

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
  vocab = np.array(sorted(set(text))) # The unique characters in the file
  np.save(checkpoint_dir+'/vocab_'+channel_name,vocab)
  vocab_size = len(vocab) # Length of the vocabulary in chars
  # Creating a mapping from unique characters to indices
  char2idx = {u:i for i, u in enumerate(vocab)}
  idx2char = np.array(vocab)


  text_as_int = np.array(String_To_Int_Vector(text,char2idx))
  training_cutoff=round(len(text_as_int)*Training_Proportion)
  train_text = text_as_int[:training_cutoff]
  validation_text = text_as_int[training_cutoff:]

  # Create training examples / targets
  dataset = tf.data.Dataset.from_tensor_slices(train_text)
  dataset = dataset.batch(seq_length+1,drop_remainder=True)
  dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(BUFFER_SIZE))
  dataset = dataset.apply(tf.data.experimental.map_and_batch(batch_size=BATCH_SIZE,drop_remainder=True,map_func=split_input_target))
  #dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

  validation_dataset = tf.data.Dataset.from_tensor_slices(validation_text)
  validation_dataset = validation_dataset.batch(seq_length+1,drop_remainder=True)
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

  examples_per_training_epoch = len(train_text)//seq_length
  steps_per_training_epoch = examples_per_training_epoch//BATCH_SIZE
  examples_per_validation_epoch = len(validation_text)//seq_length
  steps_per_validation_epoch = examples_per_validation_epoch//BATCH_SIZE
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

config_file = open(config_filename,"r")
CG_ID = config_file.readline().split()[0]
CG_password = config_file.readline().split()[0]
Chat_host = config_file.readline().split()[0]
Chat_port= config_file.readline().split()[0]
MUC = config_file.readline().split()[0]
Nickname = config_file.readline().split()[0]
Channel = config_file.readline().split()[0]
config_file.close()

if Train:
  Train_Bot(Channel,MUC)
else:
  Bot = ChannelBot(CG_ID+'@'+Chat_host,CG_password,Nickname,Channel,MUC)
  Bot.connect()
  Bot.process()