<template>
  <div>
    <v-card class="mx-auto" max-width="600">
      <v-card-title>
        <h3>Ajouter un nouvel appartement</h3>
      </v-card-title>
      <v-card-text>
        <form @submit.prevent="ajouterAppartement">
          <div>
            <v-text-field label="Nombre de chambres" v-model="nouvelAppartement.nbRooms" type="number" id="nbRooms"
              required @input="updatePredictions" />
          </div>
          <div>
            <label for="surface">Surface (m²) :</label>
            <v-text-field v-model="nouvelAppartement.surface" type="number" id="surface" required
              @input="updatePredictions" />
          </div>
          <div>
            <label for="nbWindows">Nombre de fenêtres :</label>
            <v-text-field v-model="nouvelAppartement.nbWindows" type="number" id="nbWindows" required
              @input="updatePredictions" />
          </div>
          <div v-if="predictedPrice">
            <label>Prix prédit :</label>
            <span class="predicted-value">{{ Math.round(predictedPrice) }} €</span>
          </div>
          <div v-if="predictedCategory">
            <label>Catégorie :</label>
            <span class="predicted-category">{{ predictedCategory }}</span>
          </div>
          <div v-if="predictedType">
            <label>Type :</label>
            <span class="predicted-category">{{ predictedType }}</span>
          </div>
          <div v-if="predictedNote !== null">
            <label>Note prédit :</label>
            <span class="predicted-value">{{ Math.round(predictedNote) }}</span>
          </div>
          <div v-if="predictedYear !== null">
            <label>Année prédit :</label>
            <span class="predicted-value">{{ Math.round(predictedYear) }}</span>
          </div>
          <div v-if="predictedGarage !== null">
            <label>Garage prédit :</label>
            <span class="predicted-category">{{ predictedGarage ? 'Oui' : 'Non' }}</span>
          </div>
          <div>
            <label for="customPrice">Votre prix:</label>
            <v-text-field v-model="nouvelAppartement.price" type="number" id="customPrice" required />
          </div>
          <div>
            <label for="year">Année :</label>
            <v-text-field v-model="nouvelAppartement.year" type="number" id="year" required />
          </div>
          <div>
            <label for="balcony">Balcon :</label>
            <v-checkbox v-model="nouvelAppartement.balcony" id="balcony" />
          </div>
          <div>
            <label for="garage">Garage :</label>
            <v-checkbox v-model="nouvelAppartement.garage" id="garage" />
          </div>
          <div>
            <label for="city">Ville :</label>
            <v-text-field v-model="nouvelAppartement.city" type="text" id="city" required @input="updatePredictions" />
          </div>
          <v-btn type="submit">Ajouter l'appartement</v-btn>
        </form>
      </v-card-text>
    </v-card>
  </div>
</template>

<script>
import { defineComponent, reactive, ref } from 'vue';
import axios from 'axios';

export default defineComponent({
  name: "AjoutAppartement",
  emits: ['appartement-ajoute'],
  setup(props, { emit }) {
    const nouvelAppartement = reactive({
      nbRooms: 0,
      surface: 0,
      nbWindows: 0,
      price: 0,
      year: new Date().getFullYear(), // Année actuelle par défaut
      balcony: false,
      garage: false,
      city: '',
      actions: ''
    });

    const predictedPrice = ref(0);
    const predictedCategory = ref('');
    const predictedType = ref('');
    const predictedNote = ref(null);
    const predictedYear = ref(null);
    const predictedGarage = ref(null);

    const updatePredictions = async () => {
      try {
        if (nouvelAppartement.surface > 0) {
          const priceResponse = await axios.post('http://localhost:5000/predict', {
            surface: nouvelAppartement.surface
          });
          predictedPrice.value = priceResponse.data.predicted_price;
        } else {
          predictedPrice.value = 0;
        }

        if (nouvelAppartement.surface > 0 && nouvelAppartement.nbRooms > 0 && nouvelAppartement.nbWindows > 0 && nouvelAppartement.city) {
          const categoryResponse = await axios.post('http://localhost:5000/predict-category', {
            surface: nouvelAppartement.surface,
            nbRooms: nouvelAppartement.nbRooms,
            nbWindows: nouvelAppartement.nbWindows,
            price: predictedPrice.value,
            city: nouvelAppartement.city
          });
          predictedCategory.value = categoryResponse.data.predicted_category;
        } else {
          predictedCategory.value = '';
        }

        if (nouvelAppartement.surface > 0) {
          const typeResponse = await axios.post('http://localhost:5000/predict-type', {
            surface: nouvelAppartement.surface
          });
          predictedType.value = typeResponse.data.predicted_type;
        } else {
          predictedType.value = '';
        }

        if (nouvelAppartement.surface > 0 && nouvelAppartement.price > 0 && nouvelAppartement.city) {
          const noteResponse = await axios.post('http://localhost:5000/predict-note', {
            surface: nouvelAppartement.surface,
            price: nouvelAppartement.price,
            city: nouvelAppartement.city,
            nbWindows: nouvelAppartement.nbWindows,
            nbRooms: nouvelAppartement.nbRooms
          });
          predictedNote.value = noteResponse.data.predicted_note;
        } else {
          predictedNote.value = null;
        }

        if (nouvelAppartement.price > 0 && nouvelAppartement.city) {
          const yearResponse = await axios.post('http://localhost:5000/predict-year', {
            price: nouvelAppartement.price,
            city: nouvelAppartement.city,
            surface: nouvelAppartement.surface,
            nbRooms: nouvelAppartement.nbRooms,
            nbWindows: nouvelAppartement.nbWindows
          });
          predictedYear.value = yearResponse.data.predicted_year;
        } else {
          predictedYear.value = null;
        }

        if (nouvelAppartement.price > 0 && nouvelAppartement.city) {
          const garageResponse = await axios.post('http://localhost:5000/predict-garage', {
            price: nouvelAppartement.price,
            city: nouvelAppartement.city,
            surface: nouvelAppartement.surface,
            nbRooms: nouvelAppartement.nbRooms,
            nbWindows: nouvelAppartement.nbWindows
          });
          predictedGarage.value = garageResponse.data.predicted_garage;
        } else {
          predictedGarage.value = null;
        }

      } catch (error) {
        console.error('Erreur lors de la prédiction :', error);
      }
    };

    const ajouterAppartement = () => {
      const nouvelId = Date.now();
      const appartementComplet = { ...nouvelAppartement, id: nouvelId, price: nouvelAppartement.price };
      emit('appartement-ajoute', appartementComplet);

      Object.keys(nouvelAppartement).forEach(key => {
        if (typeof nouvelAppartement[key] === 'boolean') {
          nouvelAppartement[key] = false;
        } else {
          nouvelAppartement[key] = 0;
        }
      });
      predictedPrice.value = 0;
      predictedCategory.value = '';
      predictedType.value = '';
      predictedNote.value = null;
      predictedYear.value = null;
      predictedGarage.value = null;
    };

    return { nouvelAppartement, predictedPrice, predictedCategory, predictedType, predictedNote, predictedYear, predictedGarage, ajouterAppartement, updatePredictions };
  }
});
</script>

<style scoped>
.predicted-value {
  font-weight: bold;
  color: #4caf50;
  margin-left: 10px;
}

.predicted-category {
  font-weight: bold;
  color: #2196f3;
  margin-left: 10px;
}
</style>
