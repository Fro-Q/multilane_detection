<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import axios from 'axios'
import { compileTemplate } from 'vue/compiler-sfc';

const fileURL = ref(null)
const type = ref(null)
const router = useRouter()

const fileUpload = (event) => {
  const file = event.target.files[0]
  const reader = new FileReader()

  reader.onload = (e) => {
    fileURL.value = e.target.result
  }
  reader.readAsDataURL(file)

  const fileType = file.type.split('/')[0]
  if (fileType === 'image') {
    type.value = 'image'
  } else if (fileType === 'video') {
    type.value = 'video'
  }
}

const resultBase64 = ref(null)

function submit() {
  resultBase64.value = null
  axios.
    post('/api/upload/', {
      fileURL: fileURL.value,
      fileType: type.value
    })
    .then((response) => {
      resultBase64.value = response.data.data.result
    })
    .catch((error) => {
      console.log(error)
    })
}

</script>

<template>
  <main>
    <div class="left upload">

      <div class="display-area">
        <div class="img-wrapper">
          <img :src="fileURL" alt="Uploaded image" v-if="type == 'image'" />
          <video :src="fileURL" controls v-if="type == 'video'">Your browser does not support the video tag.</video>
        </div>
      </div>
      <div class="input-area">
        <input type="file" @change="fileUpload" accept=".jpg, .jpeg, .png, .mp4" />
        <button @click="submit">Submit</button>

      </div>
    </div>
    <div class="right output">
      <div class="result">
        <div class="img-wrapper">
          <img :src="resultBase64" alt="Result image" v-if="type == 'image'" />
          <video :src="resultBase64" controls v-if="type == 'video'">Your browser does not support the video
            tag.</video>
        </div>
      </div>
    </div>
  </main>
</template>