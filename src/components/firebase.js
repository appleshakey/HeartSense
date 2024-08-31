import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import {getFirestore} from "firebase/firestore";

const firebaseConfig = {
    apiKey: "AIzaSyADvsxs-EvLw-oRWvgvF_doIZgX7pB1yGg",
    authDomain: "cardiacdiseaseriskfactor.firebaseapp.com",
    projectId: "cardiacdiseaseriskfactor",
    storageBucket: "cardiacdiseaseriskfactor.appspot.com",
    messagingSenderId: "287169903179",
    appId: "1:287169903179:web:402927d508a4105d30abce"
  };
  
  // Initialize Firebase
export const app = initializeApp(firebaseConfig);
export const auth = getAuth(app)
export const db = getFirestore(app)