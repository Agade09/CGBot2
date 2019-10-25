#!/usr/bin/env python3
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import re as regex
from slixmpp import ClientXMPP
import datetime
import os
import sys
import functools
from collections import Counter
import tensorflow as tf
from tensorflow.contrib.opt import DecoupledWeightDecayExtension,NadamOptimizer
import tensorflow.keras.backend as K

seq_length = 1000
BATCH_SIZE = 64
BUFFER_SIZE = 10000
EPOCHS=1000
embedding_dim = 64 # The embedding dimension
Is_Stateful = False
RNN_Layers = 1
rnn_units = 1024 # Number of RNN units
Training_Proportion = 0.95
load_checkpoint = True
Train = False
Log_Messages=False
Test=False
Initialisation_Length=1000
Vocab_Limit=256
Learning_Rate=1e-3
Weight_Decay=2e-4
Dropout_Rate=0.05
Early_Stopping_Patience=10 #"Number of epochs with no improvement after which training will be stopped"
# Low temperatures results in more predictable text. Higher temperatures results in more surprising text.
Max_Temperature = 1.0
Min_Temperature = 0.5
Temperature_Char_Annealing = 10
checkpoint_dir = './training_checkpoints'
logs_dir = './Logs'
config_filename = 'Config.txt'
Outputs_Dir = './Graphs'

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
  model = tf.keras.Sequential()
  #model.add(tf.keras.layers.Embedding(vocab_size,embedding_dim,batch_input_shape=[batch_size,seq_length-1]))
  model.add(tf.keras.layers.Embedding(Vocab_Limit,embedding_dim,batch_input_shape=[batch_size,seq_length-1]))#Wasteful to use Vocab_Limit but aids transfer learning from other channels
  #model.add(tf.keras.layers.Lambda(lambda x:tf.contrib.layers.group_norm(x,reduction_axes=(-3,))[0]))
  model.add(tf.keras.layers.LayerNormalization())
  #model.add(tf.keras.layers.BatchNormalization())
  #model.add(tf.keras.layers.Dropout(Dropout_Rate))
  for i in range(RNN_Layers):
    model.add(rnn(rnn_units,return_sequences=True,recurrent_initializer='glorot_uniform',stateful=Is_Stateful if Train else True))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LayerNormalization())
    model.add(tf.keras.layers.Dropout(Dropout_Rate))
  #model.add(tf.keras.layers.Dense(vocab_size))
  model.add(tf.keras.layers.Dense(Vocab_Limit))#Wasteful to use Vocab_Limit but aids transfer learning from other channels
  return model

def loss(labels, logits):
  return tf.keras.backend.sparse_categorical_crossentropy(labels,logits,from_logits=True)

class NadamWOptimizer(DecoupledWeightDecayExtension, NadamOptimizer):
  def __init__(self, weight_decay, *args, **kwargs):
    super(NadamWOptimizer, self).__init__(weight_decay, *args, **kwargs)

def Predictions_To_Id(predictions,Temperature):
  predictions = tf.squeeze(predictions, 0) # remove the batch dimension
  # using a multinomial distribution to predict the word returned by the model
  predictions = predictions / Temperature
  predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
  return predicted_id

def String_To_Int_Vector(string,char2idx):
  return np.array([(char2idx[c] if (c in char2idx) else len(char2idx)-1) for c in string],dtype=np.uint8)

def Feed_Model(model,message,char2idx):
  msg_input = String_To_Int_Vector(message,char2idx)
  msg_input = tf.expand_dims(msg_input, 0)
  predictions = model(msg_input)
  return predictions

def Current_Temperature(idx):
  return Max_Temperature+(Min_Temperature-Max_Temperature)*min(1,idx/Temperature_Char_Annealing)

def generate_response(model,start_string,idx2char,char2idx): # Evaluation step (generating text using the learned model)
  start = time.time()
  num_generate = 100 # Max Number of characters to generate
  input_eval = String_To_Int_Vector(start_string,char2idx)
  input_eval = tf.expand_dims(input_eval, 0)
  text_generated = []
  for i in range(num_generate):
      predictions = model(input_eval)
      predicted_id = Predictions_To_Id(predictions,Current_Temperature(i))
      
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
    def __init__(self, jid, password,nick,rooms,MUC_name,Ignored):
      ClientXMPP.__init__(self, jid, password)
      self.nickname=nick
      self.room_names=rooms
      self.MUC=MUC_name

      self.model = {}
      self.char2idx={}
      self.idx2char={}
      self.Ignored=Ignored
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
        text = text[len(text)-2*Initialisation_Length:]
        text = text[text.find('\n')+1:]
        text = Filter_Logs(text)
        text = text[len(text)-Initialisation_Length:]
        text = text[text.find('\n')+1:]
        Feed_Model(self.model[room_name],text,self.char2idx[room_name])#Give the model some state
      if Test:
        first_room=self.room_names[0]
        Test_Bot(self.model[first_room],self.idx2char[first_room],self.char2idx[first_room])
      self.register_plugin('xep_0045')
      self.add_event_handler("session_start", self.session_start)
      self.add_event_handler("message", self.message)
      self.add_event_handler("groupchat_message", self.muc_message)
      self.add_event_handler("session_end",self.crash)
      self.add_event_handler("socket_error",self.crash)
      self.add_event_handler("disconnected",self.crash)
      self.add_event_handler("stream_error",self.crash)
      self.add_event_handler("killed",self.crash)
      self.add_event_handler("connection_failed",self.crash)
      self.add_event_handler("close",self.crash)

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
        reply['id']=first_room+"_"+self.nickname+"@"+self.MUC+"/"+datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')+"_"+str(np.random.random())
        reply.send()

    def muc_message(self,msg):
      room_name=msg['from'].bare
      room_name=room_name[:room_name.find('@')]
      print(room_name)
      print(msg.get_mucnick()+':'+msg['body'])
      Log_Message(msg)
      Feed_Model(self.model[room_name],msg.get_mucnick()+':'+msg['body']+'\n',self.char2idx[room_name])
      if msg.get_mucnick()!=self.nickname and (self.nickname.lower() in msg['body'].lower()) and (not msg.get_mucnick() in self.Ignored):
        print("Saw my nickname")
        reply=self.make_message(mto=msg['from'].bare,mbody=generate_response(self.model[room_name],self.nickname+':',self.idx2char[room_name],self.char2idx[room_name]),mtype='groupchat')
        reply['id']=room_name+"_"+self.nickname+"@"+self.MUC+"/"+datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')+"_"+str(np.random.random())
        reply.send()

    def crash(self):
      print("crashed")
      sys.exit()

def Train_Bot(channel_name,MUC,transfer_learn_channel):
  training_start = time.time()
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
  vocab = Counter(text)
  vocab = sorted(vocab, key=vocab.get, reverse=True)
  vocab = np.array(vocab[:Vocab_Limit])
  np.save(checkpoint_dir+'/vocab_'+channel_name,vocab)
  vocab_size = len(vocab) # Length of the vocabulary in chars
  # Creating a mapping from unique characters to indices
  char2idx = {u:i for i, u in enumerate(vocab)}
  idx2char = np.array(vocab)

  Total_Text_Length = len(text)
  text=text.split('\n')
  def Pad(string,desired_length):
    if len(string)>desired_length:
      return string[:desired_length]
    else:
      return string.ljust(desired_length) #pad with spaces
  def Sample(text_by_line,desired_length=sys.maxsize,end_char='\n',target_sequences=sys.maxsize):
    text_samples=[]
    i=0
    while i<len(text_by_line) and len(text_samples)<target_sequences:
      sample=""
      while i<len(text_by_line) and len(sample)+len(text_by_line[i])+1<=desired_length: #How to deal with a really long line?
        sample+=text_by_line[i]+end_char
        i+=1;
      text_samples.append(Pad(sample,desired_length) if desired_length!=sys.maxsize else sample)
    return text_samples, i
  def Make_Stateful_Inputs(text,Total_Text_Length):
    training_length = int(np.round(Total_Text_Length*Training_Proportion))
    validation_length = Total_Text_Length-training_length
    training_sequence_length = Text_Length_To_Sequence_Lenth(training_length)
    validation_sequence_length = Text_Length_To_Sequence_Lenth(validation_length)
    print("Splitting the",training_length,"training characters of training text into",BATCH_SIZE,"sequences of",training_sequence_length,"characters each")
    print("Splitting the",validation_length,"validation characters of training text into",BATCH_SIZE,"sequences of",validation_sequence_length,"characters each")
    training_text, stop_index = Sample(text,desired_length=training_sequence_length,target_sequences=BATCH_SIZE)
    validation_text, _ = Sample(text[stop_index:],desired_length=validation_sequence_length,target_sequences=BATCH_SIZE)
    training_text = [text for text,_ in [Sample(sequence,desired_length=seq_length,end_char='') for sequence in training_text]]
    validation_text = [text for text,_ in [Sample(sequence,desired_length=seq_length,end_char='') for sequence in validation_text]]
    training_text = [seq[i] for i in range(len(training_text[0])) for seq in training_text] #[['a','b'],['1','2']] -> ['a', '1', 'b', '2']
    print('Training text converted to',len(training_text),'excerpts of length',len(training_text[0]))
    validation_text = [seq[i] for i in range(len(validation_text[0])) for seq in validation_text] #[['a','b'],['1','2']] -> ['a', '1', 'b', '2']
    return training_text, validation_text
  def Make_NonStateful_Inputs(text):
    text, _ = Sample(text,desired_length=seq_length)
    np.random.seed(0)
    np.random.shuffle(text)
    np.random.seed()
    training_cutoff=round(len(text)*Training_Proportion)
    return text[:training_cutoff], text[training_cutoff:]
  def Make_Inputs(text,Total_Text_Length):
    if Is_Stateful:
      return Make_Stateful_Inputs(text,Total_Text_Length)
    else:
      return Make_NonStateful_Inputs(text)
  def Text_Length_To_Sequence_Lenth(text_length):
    return int(np.ceil(text_length/BATCH_SIZE))
  def Accuracy(y_true,y_pred):
  	y_true = K.reshape(y_true,(BATCH_SIZE,seq_length-1))
  	y_pred = K.argmax(y_pred,axis=-1)
  	y_true = K.cast(y_true,tf.int64)
  	match = K.equal(y_true,y_pred)
  	match = K.cast(match,tf.float32)
  	return K.mean(match)
  training_text, validation_text = Make_Inputs(text,Total_Text_Length)
  print(len(training_text),len(validation_text))
  training_text_as_int = np.array([String_To_Int_Vector(s,char2idx) for s in training_text])
  validation_text_as_int = np.array([String_To_Int_Vector(s,char2idx) for s in validation_text])
  print("Training for channel",channel_name,"from",len(training_text_as_int),"samples and validating on",len(validation_text_as_int),"samples")

  # Create training examples / targets
  dataset = tf.data.Dataset.from_tensor_slices(training_text_as_int)
  if not Is_Stateful:
    dataset = dataset.shuffle(BUFFER_SIZE)
  dataset = dataset.repeat()
  dataset = dataset.map(map_func=split_input_target)
  dataset = dataset.batch(batch_size=BATCH_SIZE,drop_remainder=True)
  #dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

  validation_dataset = tf.data.Dataset.from_tensor_slices(validation_text_as_int)
  validation_dataset = validation_dataset.repeat()
  #validation_dataset = validation_dataset.batch(seq_length+1,drop_remainder=True)
  validation_dataset = validation_dataset.map(map_func=split_input_target)
  validation_dataset = validation_dataset.batch(batch_size=BATCH_SIZE,drop_remainder=True)
  #validation_dataset = validation_dataset.prefetch(tf.contrib.data.AUTOTUNE)

  model = build_model(vocab_size=len(vocab),embedding_dim=embedding_dim,batch_size=BATCH_SIZE)
  checkpoint_file = checkpoint_dir+"/weights_"+channel_name+".h5"
  transfer_learn_weights_file = checkpoint_dir+"/weights_"+transfer_learn_channel+".h5"
  if load_checkpoint and os.path.isfile(checkpoint_file):
    model.load_weights(checkpoint_file)
    print("Restarting training from previous checkpoint")
  elif transfer_learn_channel!=channel_name and os.path.isfile(transfer_learn_weights_file):
    try:
      model.load_weights(transfer_learn_weights_file)
      print("Transfer learning from channel",transfer_learn_channel)
    except:
      print("Transfer learning from channel",transfer_learn_channel,"failed to load the weights")
  else:
    print("Training from random weights")
  model.summary()
  model.compile(optimizer=NadamWOptimizer(learning_rate=Learning_Rate,weight_decay=Weight_Decay),loss=loss,metrics=[Accuracy])
  Callbacks_List = []
  checkpoint_prefix = checkpoint_dir+"/weights_"+channel_name+".h5"# Name of the checkpoint files
  Callbacks_List.append(tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,save_weights_only=True,save_best_only=True,mode='max',monitor='Accuracy'))
  if Is_Stateful:
    class LSTM_Reset_Callback(tf.keras.callbacks.Callback):
      def on_test_begin(self, batch,logs=None):
        self.model.reset_states()
      def on_epoch_begin(self, batch,logs=None):
        self.model.reset_states()
    Callbacks_List.append(LSTM_Reset_Callback())
  Callbacks_List.append(tf.keras.callbacks.EarlyStopping(patience=Early_Stopping_Patience,monitor='Accuracy',mode='max'))
  Callbacks_List.append(tf.keras.callbacks.CSVLogger(Outputs_Dir+'/training_logs_'+channel_name+'.csv', append=False, separator=';'))

  if Is_Stateful:
    assert len(training_text_as_int)%BATCH_SIZE==0,"len(training_text_as_int) not divisible by Batch Size"
    assert len(validation_text_as_int)%BATCH_SIZE==0,"len(validation_text_as_int) not divisible by Batch Size"
  steps_per_training_epoch = int(np.ceil(len(training_text_as_int)/BATCH_SIZE))
  steps_per_validation_epoch = int(np.ceil(len(validation_text_as_int)/BATCH_SIZE))
  training_history = model.fit(dataset,epochs=EPOCHS,steps_per_epoch=steps_per_training_epoch,callbacks=Callbacks_List,validation_steps=steps_per_validation_epoch,validation_data=validation_dataset)

  # summarize history for loss
  plt.yscale('log')
  plt.plot(training_history.history['loss'])
  plt.plot(training_history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.savefig(Outputs_Dir+'/loss_'+channel_name+'.png')
  plt.gcf().clear()

  plt.plot(training_history.history['Accuracy'])
  plt.plot(training_history.history['val_Accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.savefig(Outputs_Dir+'/accuracy_'+channel_name+'.png')
  plt.gcf().clear()

  print("Took",time.time()-training_start,"s to train on channel",channel_name,"'s logs")

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
Ignored_line=config_file.readline()
Ignored = Ignored_line[:Ignored_line.find("//")].split()
config_file.close()

if Train:
  for channel in Channels:
    Train_Bot(channel,MUC,Channels[0])
else:
  Bot = ChannelBot(CG_ID+'@'+Chat_host,CG_password,Nickname,Channels,MUC,Ignored)
  Bot.connect()
  Bot.process()
