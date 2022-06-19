import axios from "axios";
import { server } from "../constants/constant";

async function getDocument(token, data) {
    let response = await axios.post(server + 'document-detail', data, {
        headers: {
            authtoken: token
        },
    }).catch(function(error) {
        console.log(error)
        return false;
    });

    return response
}

async function getAnnotations(token, data) {
    let response = await axios.post(server + 'annotation-list', data, {
        headers: {
            authtoken: token
        },
    }).catch(function(error) {
        console.log(error)
        return null;
    });
    return response.data
}

async function addAnnotations(token, data) {
    let response = await axios.post(server + 'annotation', data, {
        headers: {
            authtoken: token
        },
    }).catch(function(error) {
        console.log(error)
        return false;
    });

    return response
}

async function deleteAnnotation(token, data) {
    await axios.delete(server + 'annotation', {
        headers: {
            authtoken: token
        },
        data: data,
    }).catch(function(error) {
        console.log(error)
        return false;
    });
    return true
}

async function clearPredictedAnnotations(token, data) {
    await axios.delete(server + 'clear-predicted-annotations', {
        headers: {
            authtoken: token
        },
        data: data,
    }).catch(function(error) {
        console.log(error)
        return false;
    });
    return true
}

async function trainModel(token, data) {
    let response = await axios.post(server + "train-model", data, {
        headers: {
            authtoken: token
        },
    }).catch(function(e) {
        console.log(e)
        return false;
    });
    if(response.status === 200)
        return true
    return false
}

async function getModelPools(token, data) {
    let response = await axios.post(server + "modelpool-list", data, {
        headers: {
            authtoken: token
        },
    }).catch(function(e) {
        console.log(e)
        return null;
    });
    if(response.status === 200)
        return response.data
    return null
}

async function createModelPool(token, data) {
    let response = await axios.post(server + "create-modelpool", data, {
        headers: {
            authtoken: token
        },
    }).catch(function(e) {
        console.log(e)
        return false;
    });
    if(response.status === 200)
        return true
    return false
}

async function saveModelPool(token, data) {
    let response = await axios.post(server + "change-modelpool-status", data, {
        headers: {
            authtoken: token
        },
    }).catch(function(e) {
        console.log(e)
        return false;
    });
    if(response.status === 200)
        return true
    return false
}

async function annotateDocument(token, data) {
    let response = await axios.post(server + "annotate", data, {
        headers: {
            authtoken: token
        },
    }).catch(function(e) {
        console.log(e)
        return false;
    });
    if(response.status === 200)
        return true
    return false
}

export { getDocument, deleteAnnotation, clearPredictedAnnotations, trainModel, getAnnotations, addAnnotations, getModelPools, createModelPool, saveModelPool, annotateDocument }
