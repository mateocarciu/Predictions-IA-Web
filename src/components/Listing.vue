<template>
  <div>
    <h3>Liste d'appartements</h3>
    <v-row>
      <v-col cols="12" md="8">
        <v-data-table :headers="headers" :items="suites" class="elevation-1">
          <template v-slot:item.nbRooms="{ item }">
            {{ headers }}
            <span>{{ item.nbRooms }}</span>
          </template>
          <template v-slot:item.surface="{ item }">
            <span>{{ item.surface }} m²</span>
          </template>
          <template v-slot:item.price="{ item }">
            <span>{{ item.price }} €</span>
          </template>
          <template v-slot:item.nbWindows="{ item }">
            <span>{{ item.nbWindows }}</span>
          </template>
          <template v-slot:item.year="{ item }">
            <span>{{ item.year }}</span>
          </template>
          <template v-slot:item.balcony="{ item }">
            <span>{{ item.balcony ? 'Oui' : 'Non' }}</span>
          </template>
          <template v-slot:item.garage="{ item }">
            <span>{{ item.garage ? 'Oui' : 'Non' }}</span>
          </template>
          <template v-slot:item.city="{ item }">
            <span>{{ item.city }}</span>
          </template>
          <template v-slot:item.actions="{ item }">
            <v-btn color="red" @click="supprimerAppartement(item.id)">
              Supprimer
            </v-btn>
          </template>
        </v-data-table>
      </v-col>

      <v-col cols="12" md="4">
        <add @appartement-ajoute="ajouterAppartement" />
      </v-col>
    </v-row>
  </div>
</template>

<script>
import { defineComponent, ref, onMounted } from 'vue';
import Add from './Add.vue';
import axios from 'axios';

export default defineComponent({
  components: { Add },
  setup() {
    const headers = ref[
      { text: 'Nombre de chambres', value: 'nbRooms' }
      // { text: 'Surface (m²)', value: 'surface' },
      // { text: 'Nombre de fenêtres', value: 'nbWindows' },
      // { text: 'Prix', value: 'price' },
      // { text: 'Année', value: 'year' },
      // { text: 'Balcon', value: 'balcony' },
      // { text: 'Garage', value: 'garage' },
      // { text: 'Ville', value: 'city' },
      // { text: 'Actions', value: 'actions', sortable: false }
    ];

    const suites = ref([]);

    const fetchAppartements = async () => {
      try {
        const response = await axios.get('http://localhost:3000/api/appartements');
        suites.value = response.data;
      } catch (error) {
        console.error('Erreur lors du chargement des appartements :', error);
      }
    };

    onMounted(fetchAppartements);

    const ajouterAppartement = async (nouvelAppartement) => {
      try {
        const response = await axios.post('http://localhost:3000/api/appartements', {
          id: suites.value.length + 1,
          ...nouvelAppartement
        });
        suites.value = response.data;
      } catch (error) {
        console.error('Erreur lors de l\'ajout de l\'appartement :', error);
      }
    };

    const supprimerAppartement = async (id) => {
      try {
        const response = await axios.delete(`http://localhost:3000/api/appartements/${id}`);
        suites.value = response.data;
      } catch (error) {
        console.error('Erreur lors de la suppression de l\'appartement :', error);
      }
    };
    console.log(headers);

    return { headers, suites, ajouterAppartement, supprimerAppartement };
  }
});
</script>
