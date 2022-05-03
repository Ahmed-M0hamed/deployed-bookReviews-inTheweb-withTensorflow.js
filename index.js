// let model = tf.loadGraphModel("model.json")
const tfjs_status = document.getElementById("tfjs_status");
const classes  = {
    0: 'fake' , 
    1 : 'true'
}

if (tfjs_status) {
    tfjs_status.innerText = "Loaded TensorFlow.js - version:" + tf.version.tfjs;
}

let model; // This is in global scope

const loadModel = async () => {
    try {
        const tfliteModel = await tflite.loadTFLiteModel(
        "lite_model_v1.tflite"
        );
        model = tfliteModel; // assigning it to the global scope model as tfliteModel can only be used within this scope
      // console.log(tfliteModel);

      //  Check if model loaded
        if (tfliteModel) {
            console.log('model_loaded')
        
        }
    } catch (error) {
        console.log(error);
    }
};
loadModel();


function handleButton(){
    let val = document.getElementById('review').value;
    // let val = 'ahmed is good'
    let cleaned = val.replace(/[`~!@#$%^&*()_|+\-=?;:'",.<>\{\}\[\]\\\/]/gi, '');
    let lowered = cleaned.toLowerCase()
    let splited = lowered.split(' ');
    
    const fetch_data = fetch("word_index.json")
        .then(Response => Response.json())
        .then(data => {
            let sequence_vector = [] ;
            for(i of splited){
                if(i in data) {
                    sequence_vector.push(data[i])
                    
                } else {
                    sequence_vector.push(data['UNK'])
                }
            } 
            return sequence_vector
        });

    const convert = async () => {
        let sequence_vector = await fetch_data; 
        let sequence_length = sequence_vector.length 
        let pad_length = 300 - sequence_length
        let zero_arr = Array(pad_length).fill(0)
        let padded_sequence = zero_arr.concat(sequence_vector)
        const sequence_tensor = tf.tensor2d(padded_sequence ,  [1,300])
        sequence_tensor.print()
        console.log(sequence_tensor.shape)
        const pres = model.predict(sequence_tensor)
        console.log(pres.arraySync()[0])
        prediction.textContent = classes[pres.argMax().arraySync()[0]]
    
    }
    convert()

}
