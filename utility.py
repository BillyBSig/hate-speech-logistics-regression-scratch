import re,string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import numpy as np
import pandas as pd
import math
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pickle
from time import process_time


#Logistic Regression

class LogisticRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    #Fit data X dan y
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            # approximate y with linear combination of weights and x, plus bias
            linear_model = np.dot(X, self.weights) + self.bias
            # apply sigmoid function
            y_predicted = self._sigmoid(linear_model)

            # menghitung gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    #prediksi data set yang baru
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)
    
    #Fungsi sigmoid
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    




def prediction_pipeline(text):
   return prediction_text(pickled_model, text)


#Membersihkan data dari url, tanda baca dan karakter yang tidak dibutuhkan
def textPreprocessingTokenize(text):
  text = text.lower()  # Merubah menjadi huruf kecil
  text = re.sub(r'((https?):((//)|(\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', '', text)  # remove website url
  text = re.sub(r'#(\w+)', '', text) #remove #words
  text = re.sub(r'@(\w+)', '',text) #remove @words
  text = re.sub('(pic.twitter)\w*', '', text)  # remove twitter url
  text = re.sub(f"[{string.punctuation.replace('#','')}]+", ' ', text)  # remove punctuations
  text = re.sub('[0-9]', '', text)  # remove angka
  text = re.sub('[^\x00-\x7F]+', '', text)  # remove non-ASCII karakter
  text = re.sub(r'\b\w{1,3}\b', '', text)  # remove kata yang kurang dari 3 huruf
  text = re.sub(r'\s{2,}', ' ', text)  # remove whitespaces yang berlebihan
  text = text.strip()  # remove pre atau post whitespace

  #Remove stopwords(kata hubung)
  stop_word_set = set(nltk.corpus.stopwords.words("indonesian"))
  text = ' '.join([word for word in text.split() if word not in stop_word_set])

  #Stemming data
  stemmer = StemmerFactory().create_stemmer()
  text = stemmer.stem(text) 

  #Menghapus kata entitas dan ekspresi
  entity_ekspresi = ['baswedan', 'nanies', 'ahmad', 'ridwal', 'steven', 'tjahaja', 
                   'sumarsono', 'doong', 'yaaaa', 'yeay', 'ahoknya', 'skak', 
                   'basuki', 'xoxo', 'coyy','plis', 'huuu', 'fadli', 'ajeng',
                   'sylvie', 'alexis', 'yaaa', 'haha', 'dese', 'najwa', 'sbyudhoyono', 
                   'kamil', 'koesno', 'silvi', 'farid', 'sandiaga', 'joko', 'jarot', 'Julia', 
                   'anis', 'johan', 'hihihi', 'cuihhhhhh', 'wow', 'dhani', 'nghahahaha', 'hadisurya', 
                   'hahaha', 'ahay', 'silvyy', 'agus', 'megawati', 'nuriyah', 'silvy', 'iwan', 'hook', 
                   'sylvi', 'hehhee', 'wkwkwk', 'sinta', 'sudradjat', 'ganjar', 'dehh', 'aahhh', 
                   'ahokdjarot', 'anies', 'sulistyo', 'idih', 'pranowo', 'sandy', 'djarot', 'ahok', 'haduhhhh', 'gubraaaaak','bwahaha']
  text = ' '.join([word for word in text.split() if word not in entity_ekspresi])


  #Koreksi kata yang memiliki kesalahan penulisan dan kata serapan
  koreksi_dict = {'ngumpet':'sembunyi', "ngambang" : "ambang", "ngantuk" : "kantuk", "wjar":"wajar", "ngimpi":"mimpi", "pdhl":"padahal", 'bangsatin':'bangsat', 
                'baekin':'baik', 'dapurr':'dapur',  'ajah':'saja', 'pitnah':'fitnah', 'bnyak' : 'banyak',  'denger' : 'dengar', 'merka' : 'mereka', 'degan' : 'dengan',
                'dibully' : 'rundung', 'dtrma' : 'terima', 'mrka' : 'mereka', 'settingan' : 'atur', 'mendulungmu' : 'dukung', 'ilklan' : 'iklan', 'sbgai' : 'sebagai', 
                'kejakarta' : 'Jakarta', 'anjingin' : 'anjing', 'maren' : 'kemarin', 'baper' : 'rasa','biruadanya' : 'biru', 'telaaaakkk' : 'telak', 'ngawur' :  'kacau', 
                'pngn ': 'ingin', 'erti' : 'arti', 'ngapain' : 'apa', 'percya' : 'percaya', 'bangett'  : 'banget', 'tlah' : 'telah','nagih' : 'tagih','nbela' : 'bela', 
                'memkasakan' : 'paksa', 'kluarga' : 'keluarga', 'asbun' : 'asal', 'administrator' : 'administrasi', 'menfitnah' : 'fitnah', 'liwat' : 'lewat',
                'bhasil': 'berhasil','mlyani' : 'melayani', 'asuuu' :'anjing','prog' : 'program', 'klau':'kalau', 'cumin':'cuma', 'semalem' : 'malam', 'ngaco' : 'kacau', 
                'bagaiamana' :'bagaimana', 'kreta' :'kereta',  'ngaruh' : 'pengaruh', 'ngadu' : 'adu', 'moga' : 'semoga', 'gubenur' : 'gubernur', 'skrng' :'sekarang',  
                'mantab' : 'mantap', 'ngejek'  : 'ejek', 'respech' : 'respek', 'sorga' : 'surga',  'makmak' : 'ibu', 'ngefans' : 'idola', 'blon' : 'belum', 'team' : 'tim', 
                'mksd' : 'maksud', 'nyesel' : 'sesal',  'knpa' : 'kenapa', 'facrbook' : 'facebook', 'manies' : 'manis',  'pacarin' : 'pacar', 'dituduhsbg' : 'tuduh',  
                'neng' : 'dik','sblm' : 'sebelum','nyewa' : 'sewa', 'mantapppp' : 'mantap','sdgkan' : 'sedang','ngomen' : 'kometar', 'sibabi' : 'babi','kaleeeee' : 'kali', 
                'ntinggal' :'tinggal', 'smua' : 'semua', 'grombolannya' : 'kelompok','surveynya' : 'survey', 'pantes' : 'pantas','pemaparantentang' : 'papar', 'digituin' : 'itu',
                'gausa' : 'tidak', 'djan' : 'jangan', 'ntar' : 'nanti','tuntuna' : 'tuntun', 'aamiin': 'amin', 'yangg' : 'yang', 'terurama' : 'utama', 'asong' : 'asing', 
                'iklas' : 'ikhlas', 'anjir' : 'anjing', 'penfitnah' : 'fitnah','kaga' : 'tidak', 'saludd' : 'salut', 'gede' : 'besar', 'nselain' : 'selain', 'rasain' : 'rasa', 
                'mampuuuusssssss' : 'mampus', 'biadap' : 'biadab', 'ndendam' : 'dendam', 'kafr' : 'kafir', 'gantiin' : 'gantu', 'wktunya' : 'waktu', 'masyarkt' : 'masyarakat',
                'wlaupun' : 'walau', 'kras' : 'keras', 'kite' : 'kita', 'bleh': 'boleh', 'drpd' : 'daripada', 'ngatasi' : 'atas', 'ngurus' : 'urus', 'perempuannn' : 'perempuan', 
                'memporakporandakan' : 'porak poranda',  'dijadiin' : 'jadi', 'yaiyalah' : 'iya','lansung' : 'langsung', 'bnyk' : 'banyak', 'faham' : 'paham', 'china' : 'cina',
                'ngaji' : 'kaji', 'maluin' : 'malu', 'semagat' : 'semangat', 'mnding' : 'mending', 'kalo' : 'kalau', 'ntdk' : 'tidak', 'nyoblos' : 'coblos','radikalisme' : 'radikal', 
                'trobosan' : 'terobosan', 'penghiyanat' : 'khianat', 'skrg' : 'sekarang', 'ngeles' : 'hinder',  'stiap' : 'setiap', 'bhan' : 'bahan', 'dar' : 'dari', 
                'nampaknya' : 'tampak', 'aminnn' : 'amin', 'pemimpim' : 'pimpin','blusukan' : 'masuk', 'ngepret' : 'kepret', 'memposting' : 'posting', 'ngmg' : 'ucap', 'siapaaaa' : 'siapa', 
                'didoain' : 'doa', 'mikir' : 'pikir', 'asiang' : 'asing', 'ngoceh' : 'oceh', 'ntidak' : 'tidak', 'nguntungin' : 'untung', 'dibiarin' : 'biar', 'inget' : 'ingat', 'pemenagnya' : 'pemenang', 
                'mpok' : 'kakak', 'junjunganya' : 'junjung', 'emang' : 'memang', 'sbnrnya' : 'benar', 'ayoo' : 'ayo', 'seorg' : 'orang', 'ngamuk' : 'amuk',
                'ngomongin' : 'ucap', 'informal' : 'info', 'prnh' : 'pernah', 'ente' : 'kau', 'bego' : 'bodoh', 'dngn' : 'dengan', 'nyambung' : 'sambung', 'ngurusi' : 'urus', 
                'commentnya' : 'komentar', 'mjadi' : 'jadi', 'ngambil' : 'ambil','slama' : 'lama','jahattttt' : 'jahat', 'ngebelain' : 'bela','maen' : 'main','ngakak' : 'lucu', 
                'ingetin' : 'ingat','ditinggiin' : 'tinggal', 'trima' : 'terima', 'ateknya' : 'antek', 'sukurin' : 'syukur', 'milih' : 'pilih','ngerasain' : 'rasa', 'ceburin' : 'cebur', 
                'nanaknya' : 'anak', 'emakemak' : 'ibu', 'balikin' : 'abik', 'makanye' : 'maka','diaa' : 'dia', 'napa' : 'kenapa','bkerja' : 'bekerja', 'pasuk' : 'pasukan', 
                'sepenuh' : 'penuh', 'gendheng' : 'gila', 'akuh' : 'aku','bang' : 'abang', 'ngerti' : 'paham', 'make' : 'pakai', 'isyu' : 'isu', 'nyambungnya' : 'sambung', 
                'aduuuuuuh' : 'aduh', 'dripada' : 'dari', 'cuapcuap' : 'ucap', 'elu' : 'kau', 'prustasi' : 'frustasi', 'agam' : 'agama','nggak' : 'tidak', 'salamjari' : 'salam', 
                'akitbat' : 'akibat', 'ngaku' : 'aku','mampuuss' : 'mampus', 'satuuuu' : 'satu', 'bloonnya' : 'bodoh', 'ngetweet' : 'tweet','diporakporandakan' : 'porak-poranda', 
                'untk' : 'untuk', 'tetep' : 'tetap','nunduk' : 'tunduk', 'disiapin' : 'siap', 'jongon' : 'jangan', 'sesorng' : 'orang', 'nmengapa' : 'mengapa',
                'pakkk' : 'pak', 'idolain' : 'idola', 'kalik' : 'kali','nyuruh' : 'suruh','mlulu' : 'melulu','dijakarta' : 'Jakarta', 'kslian' : 'kalian','prilaku' : 'perilaku',
                'nyerang' : 'serang','ngomong' : 'omong', 'slalu' : 'selalu','gurbernur': 'gubernur'}
  text = ' '.join(str(koreksi_dict.get(word, word)) for word in text.split())
  
  #Tokenisasi data
  text = nltk.tokenize.word_tokenize(text) 
  return text


word_set = {'pusing', 'genang', 'beli', 'mantap', 'daan', 'sumber', 'penuh', 'mulut', 'dalam', 'kau', 'gampang', 'ayom', 
            'telak', 'bincang', 'hayat', 'prestasi', 'terimakasih', 'waras', 'agama', 'kamboja', 'sedih', 'aduh', 'negara', 
            'siuman', 'kompeten', 'nikmat', 'sentil', 'demonstran', 'poranda', 'tutup', 'wkwk', 'pecat', 'tanah', 'gantu', 
            'atas', 'ganti', 'datar', 'rubah', 'maaf', 'sentimen', 'siang', 'jamban', 'beneran', 'rukun', 'antek', 'borok', 
            'selatan', 'encer', 'frustasi', 'gadungan', 'tuntut', 'curang', 'pandang', 'untung', 'pulau', 'panggil', 'muhammad', 
            'dangdut', 'juara', 'dugem', 'rupiah', 'indonesia', 'lahan', 'tinggal', 'daya', 'artikel', 'rusuh', 'ngasih', 'serius', 
            'pake', 'rasulullah', 'ekonomi', 'tuhan', 'lontar', 'memang', 'aktif', 'tulis', 'nebar', 'bidara', 'cepat', 'julia', 
            'reklamasi', 'akhlak', 'babi', 'slip', 'rusak', 'junjung', 'cacimaki', 'parpol', 'beliau', 'belikan', 'tahu', 'lawan', 
            'moyang', 'busuk', 'nafsu', 'tentu', 'mari', 'telah', 'weekend', 'jelang', 'ahoax', 'asuh', 'jejak', 'asik', 'harga', 
            'iklan', 'ahokers', 'palestina', 'hadeuuuh', 'vonis', 'tokoh', 'kapolda', 'imam', 'tular', 'best', 'spanduk', 'cacing', 
            'dajjal', 'bagus', 'tampak', 'aming', 'berangkat', 'sipit', 'istana', 'ngga', 'onta', 'please', 'bersih', 'letak', 'cuit', 
            'kategori', 'tari', 'belum', 'omong', 'ujar', 'bubur', 'rangka', 'sedia', 'wujud', 'belia', 'pesan', 'tuai', 'real', 
            'mbak', 'perilaku', 'ada', 'timpang', 'rekam', 'gusur', 'mata', 'langsung', 'lantik', 'impas', 'bangkit', 'kalah', 
            'pasukan', 'gagas', 'sambung', 'pagi', 'abang', 'lanjur', 'desak', 'bangsa', 'ibu', 'bajing', 'riau', 'pernah', 'pak', 
            'nasional', 'keras', 'bool', 'batin', 'udah', 'hibur', 'maki', 'sampe', 'karam', 'brimob', 'familiar', 'limpah', 'trauma', 
            'syariat', 'bunuh', 'jatuh', 'puluh', 'serbet', 'allah', 'berani', 'sipenista', 'transportasi', 'gitu', 'gara', 'lebar', 
            'jilbab', 'doa', 'tuan', 'cepet', 'surga', 'akun', 'layan', 'garong', 'tahun', 'lancar', 'integrasi', 'tunjuk', 'ajang', 
            'sepatu', 'army', 'arogan', 'munafik', 'larang', 'mulia', 'stigma', 'sistem', 'kayak', 'botak', 'alhamdulillah', 'kasih', 
            'karakter', 'seko', 'malteng', 'iblis', 'ambulans', 'tonton', 'personel', 'sesat', 'tonggak', 'karya', 'jawab', 'cuman', 
            'diam', 'nasrani', 'kutik', 'betawi', 'radikal', 'kampung', 'minggu', 'dari', 'ibukota', 'tebar', 'kedok', 'tiap', 
            'kapolri', 'konsisten', 'kakak', 'wanita', 'cerdas', 'unfollow', 'baik', 'semua', 'kredit', 'waspada', 'bumi', 
            'rejekinya', 'nonton', 'hinder', 'vote', 'hukum', 'koplo', 'bersin', 'sambut', 'kebih', 'main', 'atur', 'hujat', 'pasti', 
            'tamu', 'pribumi', 'ejek', 'kalian', 'dunia', 'jajan', 'sungai', 'bandung', 'bodoh', 'kapabilitas', 'more', 'wartawan', 
            'kali', 'keruk', 'janji', 'cililitan', 'korban', 'kutuk', 'malaikat', 'brosur', 'tindak', 'dingin', 'tabur', 'bangun', 
            'nyaman', 'rezeki', 'akuang', 'pamor', 'toleransi', 'makna', 'narasi', 'jago', 'khusus', 'mendagri', 'gusti', 'uppercut', 
            'duel', 'duduk', 'disholatin', 'suap', 'nista', 'sosmed', 'tionghoa', 'jabat', 'kena', 'tunggu', 'ngotot', 'pemda', 
            'banjir', 'malam', 'domisili', 'kumpul', 'jebak', 'jilid', 'elektabilitas', 'nenek', 'menteri', 'domba', 'gorong', 
            'cerah', 'leceh', 'adjat', 'les', 'harap', 'perez', 'gereja', 'sholat', 'murtadz', 'bahaya', 'aspal', 'promosi', 'besar', 
            'ingat', 'citra', 'buka', 'paket', 'media', 'konstitusi', 'mundur', 'curhat', 'teman', 'tembak', 'alloh', 'melayani', 
            'sopan', 'sandi', 'world', 'calon', 'sadar', 'santun', 'artis', 'pasar', 'modal', 'ras', 'jijik', 'untuk', 'jumat', 
            'teladan', 'agree', 'pluit', 'hajar', 'tour', 'suci', 'maksud', 'overdosis', 'survei', 'contoh', 'kandidat', 'juang', 
            'pakai', 'arti', 'hahahahhahaa', 'ucap', 'phobia', 'morfotin', 'lama', 'angkat', 'pinter', 'gagap', 'tahap', 'ayo', 
            'nilai', 'cermin', 'dalem', 'manis', 'bayar', 'angin', 'seru', 'ikhlas', 'kasus', 'sara', 'info', 'panas', 'penjara', 
            'komplit', 'dukung', 'selalu', 'serentak', 'kaya', 'engga', 'nurani', 'ahhh', 'mereka', 'jakbar', 'campaign', 'kuasa', 
            'mujur', 'bungkus', 'lepas', 'perang', 'paham', 'kompensasi', 'sibuk', 'kecuali', 'salah', 'perhati', 'jawara', 'paslon', 
            'rame', 'gilir', 'bener', 'drama', 'rezim', 'bareng', 'aman', 'marah', 'kemendagri', 'jamin', 'perintah', 'ramai', 'rek', 
            'cina', 'salam', 'ragu', 'lambat', 'tahan', 'menang', 'rencana', 'solusi', 'libat', 'dengan', 'keluarga', 'licik', 
            'semoga', 'wajar', 'rindu', 'wangi', 'astagfirullah', 'intimidasi', 'makan', 'bebas', 'congornyapecah', 'maju', 'nelayan', 
            'struktur', 'kontestasi', 'jlebb', 'asing', 'pilgub', 'korea', 'anak', 'emank', 'senjata', 'kiri', 'jiwa', 'hindar', 
            'tangguh', 'oplos', 'sewa', 'dakwa', 'lantai', 'daerah', 'asa', 'butuh', 'terobosan', 'bom', 'singgung', 'korupsi', 'kita', 
            'akal', 'bekerja', 'zikir', 'popular', 'jalan', 'sombong', 'ogah', 'mahfud', 'coblos', 'risiko', 'nusantara', 'liat', 
            'dzalim', 'uji', 'hina', 'siap', 'hebat', 'olok', 'anti', 'intonasi', 'durjana', 'sakit', 'segi', 'buat', 'amat', 'tarik', 
            'peluang', 'pasang', 'kabul', 'provinsi', 'hastag', 'abai', 'judul', 'miskin', 'jelek', 'patut', 'cabe', 'lelah', 
            'konfrensi', 'people', 'maut', 'otak', 'muak', 'lindung', 'program', 'banget', 'antar', 'mudah', 'kacau', 'kapling', 
            'oke', 'pokok', 'akan', 'prefer', 'ungkap', 'mimpi', 'aparatur', 'gaza', 'hoax', 'sokong', 'nama', 'kesi', 'tuips', 
            'gimana', 'ajar', 'salut', 'batas', 'ilustrasi', 'shame', 'depresi', 'presiden', 'simpul', 'urusin', 'suka', 'kaliiii', 
            'ketutunan', 'reparasi', 'mau', 'pasu', 'adlh', 'Jakarta', 'koruptor', 'tani', 'bingung', 'resah', 'goblok', 'cyber', 
            'cuih', 'fitnah', 'engkau', 'ahoak', 'apps', 'makasih', 'apkabar', 'lomba', 'data', 'coba', 'ulah', 'akibat', 'sesal', 
            'tidak', 'visi', 'duga', 'serang', 'ganyang', 'bain', 'laut', 'kerja', 'kota', 'republika', 'yang', 'oceh', 'publik', 
            'selesai', 'rscm', 'kitab', 'turun', 'ulama', 'biar', 'muka', 'parasit', 'badj', 'tengah', 'dan', 'sabar', 'adil', 
            'jujur', 'sidang', 'kati', 'gera', 'kesatria', 'berita', 'kelompok', 'sepakbola', 'baiat', 'belenggu', 'tonjok', 
            'susul', 'berat', 'tagih', 'terorist', 'uang', 'normal', 'ambil', 'pecah', 'abdi', 'urus', 'tiga', 'karna', 'congor', 
            'pahala', 'budak', 'asli', 'usung', 'sungguh', 'cengar', 'maling', 'versi', 'bopeng', 'setiap', 'kompas', 'kemarin', 
            'hasut', 'sebut', 'delete', 'jambi', 'sikap', 'foto', 'tanya', 'luka', 'jual', 'aktifitas', 'integritas', 'wilayah', 
            'hasil', 'kutil', 'najis', 'takut', 'fakta', 'kredibel', 'sejahtera', 'biadab', 'tengkuk', 'tau', 'april', 'muslim', 
            'forever', 'bawaslu', 'taubat', 'sayap', 'bantuin', 'ahox', 'apbd', 'nunukan', 'sang', 'jasad', 'unggul', 'online', 
            'sekolah', 'bal', 'daging', 'ilusi', 'antem', 'bahan', 'jakarta', 'utuh', 'mateng', 'idiot', 'cerita', 'hookk', 'pusat', 
            'administratur', 'lihat', 'oktober', 'taik', 'intelektual', 'manusiawi', 'nkri', 'kyai', 'rendah', 'sekarang', 'sarap', 
            'mantan', 'lisan', 'lanjut', 'ngurusin', 'kelojot', 'pihak', 'tua', 'kaji', 'komunis', 'islam', 'abis', 'bilang', 
            'sukses', 'mending', 'kompak', 'suara', 'korup', 'single', 'silah', 'rela', 'tenaga', 'anjing', 'dagang', 'saksi', 
            'statement', 'bahagia', 'mboten', 'empati', 'realisasi', 'pemenang', 'mampir', 'wawancara', 'negeri', 'socmed', 
            'lengkap', 'metode', 'istri', 'share', 'seyogyanya', 'lgbt', 'didik', 'ticketing', 'bangsat', 'lapang', 'kriminalisasi', 
            'ubah', 'bisa', 'polres', 'penting', 'sumpah', 'kaum', 'hyung', 'putus', 'semangat', 'debat', 'sorot', 'sudut', 'dua', 
            'gesa', 'jaga', 'musik', 'serap', 'tuduh', 'orang', 'pulang', 'rangkul', 'caci', 'sembarang', 'cuti', 'ahoker', 
            'agitatif', 'sejarah', 'kereta', 'titik', 'klompok', 'buta', 'lahir', 'palsu', 'hormat', 'mancanegara', 'pers', 
            'hitung', 'tim', 'bangga', 'akuntabel', 'sare', 'andai', 'mati', 'spin', 'verbal', 'bikin', 'donor', 'demen', 'ikut', 
            'bidang', 'tweet', 'saja', 'hubung', 'aceh', 'gembira', 'jabar', 'intelek', 'raya', 'pikir', 'bekas', 'djarotlah', 
            'dibriefing', 'ambang', 'sahabat', 'pasien', 'tumpul', 'dasar', 'kuat', 'jari', 'cipta', 'itu', 'sengaja', 'cino', 
            'warga', 'bani', 'suguh', 'isu', 'anggar', 'proses', 'heran', 'cewe', 'ribut', 'mecin', 'banyak', 'masjid', 'iman', 
            'kail', 'wkwkwkwkwkk', 'kapolres', 'pngn', 'niat', 'ringsek', 'juru', 'utama', 'gontor', 'mangga', 'porak', 'wajib', 
            'misi', 'figur', 'teror', 'kelan', 'poligami', 'gelar', 'perempuan', 'brand', 'invest', 'bedain', 'pertiwi', 'islami', 
            'habis', 'performa', 'kejam', 'komitmen', 'injek', 'ketemu', 'bangke', 'jawa', 'usaha', 'mudahhan', 'ngebacot', 'polri', 
            'fatwa', 'doank', 'abik', 'boleh', 'burung', 'level', 'juta', 'mutu', 'manusia', 'trump', 'cengir', 'simak', 'rahmat', 
            'sylviana', 'pantas', 'nyata', 'tukang', 'dosen', 'partai', 'pegang', 'doang', 'benci', 'nyinyir', 'ungkit', 'closing', 
            'nasdem', 'urbanisasi', 'keluar', 'aseng', 'baju', 'baca', 'bicara', 'pale', 'puji', 'meridhoi', 'simpen', 'sepi', 'kaca', 
            'kecewa', 'februari', 'mogot', 'langkah', 'timbang', 'gaplok', 'litbang', 'pemuda', 'metro', 'bahas', 'adab', 'triliun', 
            'tangkap', 'enak', 'terima', 'tanggal', 'nada', 'syukur', 'kalimat', 'suruh', 'kang', 'habitat', 'kebhinekaan', 'kah', 
            'tuntun', 'persis', 'yakin', 'jahat', 'pegawai', 'kirim', 'didatengin', 'gabisa', 'pribadi', 'idola', 'copot', 'tetap', 
            'bara', 'sesuai', 'murni', 'balas', 'tempuh', 'polling', 'nomor', 'jadi', 'telor', 'skakmat', 'putar', 'pemprov', 'kesel', 
            'gagal', 'bongkar', 'alat', 'hidup', 'pora', 'aniessandiuno', 'tentram', 'malu', 'sholatin', 'walau', 'banser', 'mohon', 
            'jika', 'aku', 'sapa', 'some', 'antusias', 'papar', 'waduk', 'ibadah', 'peduli', 'blbi', 'politik', 'alami', 'hantu', 
            'acara', 'santri', 'wagub', 'trans', 'potong', 'facebook', 'mampus', 'posisi', 'protes', 'haram', 'tolong', 'izin', 
            'kagum', 'kejar', 'hilang', 'daripada', 'kampret', 'tulus', 'mayoritas', 'pancasila', 'baru', 'bukti', 'sampah', 'leher', 
            'cantik', 'khianat', 'puas', 'infrastruktur', 'islamophobia', 'depan', 'twit', 'ateis', 'curiga', 'gabung', 'selip', 
            'mulu', 'satu', 'janda', 'cekak', 'mafia', 'setia', 'suasana', 'kapal', 'utang', 'terjun', 'sembunyi', 'deklarasi', 
            'datang', 'family', 'pimpin', 'musnah', 'format', 'novel', 'quick', 'gin', 'sektor', 'tangan', 'cinta', 'jokopret', 
            'cemilan', 'tenang', 'rapih', 'plus', 'perna', 'damai', 'tolol', 'kristen', 'nang', 'tuju', 'hati', 'timur', 'putra', 
            'well', 'lacur', 'lewat', 'cela', 'realistis', 'birokrasi', 'rakyat', 'apa', 'pilih', 'tingkat', 'aneh', 'sayang', 
            'keukeuh', 'napi', 'pergubnya', 'gubernur', 'ambisius', 'nongkrongi', 'nyebar', 'hancur', 'tarif', 'balut', 'pantura', 
            'count', 'sebar', 'adzab', 'tindas', 'kecut', 'warna', 'sing', 'emoh', 'selamat', 'kenapa', 'dik', 'gaji', 'rombak', 
            'gila', 'anggap', 'kelar', 'istiqlal', 'sangka', 'kami', 'susu', 'ganggu', 'bacot', 'makhluk', 'laku', 'diem', 'tipis', 
            'sesi', 'gambir', 'sehat', 'kasi', 'fanboy', 'paling', 'bawa', 'duit', 'rumah', 'topeng', 'pintar', 'percaya', 'energi', 
            'provokasi', 'cape', 'dangkal', 'responden', 'jokower', 'bual', 'belah', 'murtad', 'puki', 'bohong', 'bless', 'bangkrut', 
            'bombardir', 'kerudung', 'milik', 'ronde', 'dekat', 'depok', 'iya', 'masuk', 'konyol', 'wakil', 'rasa', 'purnama', 'surat', 
            'erka', 'sinergi', 'ruku', 'tobat', 'skema', 'vital', 'kubu', 'sampel', 'sisip', 'gembel', 'komit', 'lejit', 'dia', 'naik', 
            'beritasatu', 'stres', 'posting', 'saudara', 'awal', 'video', 'kalem', 'works', 'beda', 'ketat', 'ampun', 'banding', 
            'cagub', 'langgar', 'payah', 'halal', 'cakep', 'buruk', 'kawal', 'perban', 'koar', 'metoda', 'setan', 'pulkada', 'medsos', 
            'komentar', 'beranda', 'tempat', 'sosok', 'kait', 'tipu', 'kenal', 'emosi', 'emut', 'wahabi', 'gaya', 'akademisi', 'elus', 
            'buzzer', 'masyarakat', 'amin', 'layak', 'murah', 'temu', 'tugas', 'gaduh', 'bahasa', 'temen', 'waktu', 'sempat', 'pesta', 
            'cawgub', 'ketidakadilan', 'syariah', 'badja', 'begitu', 'mang', 'lampung', 'bulat', 'nanti', 'asal', 'agen', 'maidah', 
            'tetangga', 'besok', 'kalau', 'henti', 'kampanye', 'oknum', 'lucu', 'jangan', 'kubur', 'twitter', 'bangkang', 'black', 
            'tanggung', 'kelas', 'sadis', 'polisi', 'alias', 'jasa', 'nila', 'polrestro', 'woww', 'survey', 'status', 'watak', 'nasi', 
            'paksa', 'kondisi', 'audio', 'barusan', 'basis', 'kenyataanya', 'mana', 'prediksi', 'nunggu', 'sebelum', 'happy', 
            'narkoba', 'kicau', 'balaikota', 'kafir', 'aksi', 'dapur', 'final', 'orangtua', 'demokrasi', 'jokowi', 'sebarin', 'keren', 
            'bbrp', 'umat', 'bela', 'lupa', 'pilkada', 'baginda', 'gerombol', 'peluk', 'kasar', 'kurang', 'love', 'blunder'}

total_document = 491

#Membuat count dictionary
def count_dict(sentences):
  frequency = []
  for i in range(len(sentences)):
    word_count = {}
    for word in word_set:
        word_count[word] = 0
        for sent in sentences[i]:
            if word in sent:
                word_count[word] += 1
    frequency.append(word_count)
  return frequency


#menghitung Term Frekuensi tiap kalimat
def computeTF(wordDict, doc):
  tf_collect = []
  for i in range(len(wordDict)):
    tfDict = {}
    corpusCount = len(doc[i])
    for word, count in wordDict[i].items():
        tfDict[word] = count/float(corpusCount)
    tf_collect.append(tfDict)
  return(tf_collect)


#Menghitung Inverse Document Frequency tiap kalimat
def computeIDF(docList):
    idfDict = {}
    N = total_document
    
    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for word, val in idfDict.items():
        idfDict[word] = math.log10(N / (float(val) + 1))
    return(idfDict)


#Menghitung TFIDF tiap kalimat
def computeTFIDF(tfBow, idfs):
  tfidf = []
  for i in range(len(tfBow)):
    tfidf_dict = {}
    for word, val in tfBow[i].items():
        tfidf_dict[word] = val*idfs[word]
    tfidf.append(tfidf_dict)
  return(tfidf)


#Fungsi kalkulasi tfidf
def tfidf_process(text_token):
  sentences_test = np.expand_dims(text_token, axis=0).tolist()
  freq_collect_test = count_dict(sentences_test)
  term_freq_test = computeTF(freq_collect_test, text_token)
  idfs_test = computeIDF(freq_collect_test)
  tfidf_test = computeTFIDF(term_freq_test, idfs_test)
  tfidf_test_df = pd.DataFrame(tfidf_test)

  return tfidf_test_df

def prediction_text(data, model):
  t1_start = process_time()
  test_text = textPreprocessingTokenize(data)
  tfidf_test = tfidf_process(test_text)
  prediction = model.predict(tfidf_test)
  if prediction == 1:
    output =  "Hate Speech"
  else:
    output =  "Non Hate Speech"
  t1_stop = process_time()
  result = {"data": {'prediction' : output, "text": data}, 
              "description": "Text to be predicted",
              "processing_time": f"{t1_stop - t1_start} seconds"}
  
  return result


if __name__=="__main__":
    
    with open('api/model/LogisticReg.pkl', 'rb') as f:
       pickled_model = pickle.load(f)
    textHS = "Betul bang hancurkan merka bang, musnahkan china babi dibumi pertiwi indonesia, berkedok reklamasi itu ahok"
    
    predictions = prediction_text(textHS, pickled_model)
    print(predictions)
