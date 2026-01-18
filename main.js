// frontend/main.js
const API = (window.location.hostname === '127.0.0.1' || window.location.hostname === 'localhost')
  ? 'http://127.0.0.1:8000'   // dev
  : 'https://YOUR_BACKEND_URL'; // replace after deploy

const btn = document.getElementById('predictBtn')
const clearBtn = document.getElementById('clearBtn')
const lyricsEl = document.getElementById('lyrics')
const statusEl = document.getElementById('status')
const resultBox = document.getElementById('result')
const genreEl = document.getElementById('genre')
const confEl = document.getElementById('confidence')
const scoresEl = document.getElementById('scores')

function setStatus(t){ statusEl.textContent = t }

clearBtn.onclick = () => {
  lyricsEl.value = ''
  resultBox.classList.add('hidden')
}

btn.onclick = async () => {
  const text = lyricsEl.value.trim()
  if(!text){ alert('Paste some lyrics first'); return }
  setStatus('Predicting...')
  btn.disabled = true
  try{
    const resp = await fetch(`${API}/predict`, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({text})
    })
    if(!resp.ok){
      const txt = await resp.text()
      throw new Error(`API error ${resp.status}: ${txt}`)
    }
    const data = await resp.json()
    showResult(data)
  }catch(err){
    alert('Error: ' + err.message)
    console.error(err)
  }finally{
    setStatus('')
    btn.disabled = false
  }
}

function showResult(data){
  resultBox.classList.remove('hidden')
  genreEl.textContent = data.predicted_genre
  confEl.innerHTML = `<small>Confidence: ${(data.confidence*100).toFixed(2)}%</small>`
  scoresEl.innerHTML = ''
  const pairs = Object.entries(data.scores || {})
  // sort descending
  pairs.sort((a,b)=> b[1]-a[1])
  for(const [label, p] of pairs){
    const row = document.createElement('div')
    row.className = 'score-row'
    const left = document.createElement('div'); left.textContent = label
    const right = document.createElement('div'); right.innerHTML = `${(p*100).toFixed(2)}%`
    const barWrap = document.createElement('div'); barWrap.style.width = '100%';
    const bar = document.createElement('div'); bar.className = 'bar'
    const fill = document.createElement('div'); fill.className = 'fill'
    fill.style.width = Math.round(p*100) + '%'
    bar.appendChild(fill)
    const container = document.createElement('div')
    container.style.display = 'flex'; container.style.gap = '12px'; container.style.alignItems='center'
    container.appendChild(bar)
    row.appendChild(left)
    row.appendChild(right)
    scoresEl.appendChild(row)
    scoresEl.appendChild(container)
  }
}
