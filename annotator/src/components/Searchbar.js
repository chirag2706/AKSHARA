import React, { useState, useEffect } from "react";
import { connect } from "react-redux";
import useFullPageLoader from "../hooks/useFullPageLoader";
import "../styles/documentCard.scss";
import "../styles/searchBar.scss";
import { useHistory } from "react-router-dom";
import { useParams } from "react-router-dom";
import { getUserProjects } from "../scripts/project";
import Sidebar from "./Sidebar";
import Uploaddocumentmodal from "./modals/Uploaddocumentmodal";
import { getDocuments, uploadDocument } from "../scripts/documents";
import { fetchAllAnnotationsOfAParticularProject } from "../scripts/annotation";

function Card(props) {
    return (
        <div
            className="card"
            onClick={() => props.handleDocumentClick(props.id)}
        >
            <div className="card__body">
                <img src={props.img} alt="" className="card__image" />
                <h2 className="card__title">{props.title}</h2>
                <p className="card__description">{props.description}</p>
            </div>
            {props.is_annotated ? (
                <button className="card__btn">Annotated Document</button>
            ) : (
                <button className="card__btn">Non Annotated Document</button>
            )}
        </div>
    );
}

let SearchBar = (props) => {
    let history = useHistory();
    if (!props.auth.token) {
        history.push("/login");
    }

    const [APIData, setAPIData] = useState([]);
    const [AnnotationData, setAnnotationData] = useState([]);
    const [filteredResults, setFilteredResults] = useState([]);
    const [searchInput, setSearchInput] = useState("");
    const [loaderComponent, showLoader, hideLoader] = useFullPageLoader();

    const { projectId } = useParams();
    const [ownedProjects, setOwnedProjects] = useState(null);
    const [sharedProjects, setSharedProjects] = useState(null);
    const [pageStatus, setPageStatus] = useState("1");

    const [uploadingDocument, setUploadingDocument] = useState(false);
    const handleUploadDocument = async () => {
        let title = document.getElementById("uploadDocumentTitleInput").value;
        let description = document.getElementById(
            "uploadDocumentDescriptionInput"
        ).value;
        let image = document.getElementById("uploadDocumentImageInput").files;
        if (title.split(" ").join("") === "") {
            document.getElementById(
                "uploadDocumentTitleHelperText"
            ).style.display = "inline";
        } else if (description.split(" ").join("") === "") {
            document.getElementById(
                "uploadDocumentDescriptionHelperText"
            ).style.display = "inline";
        } else if (image.length === 0) {
            document.getElementById(
                "uploadDocumentImageHelperText"
            ).style.display = "inline";
        } else {
            document.getElementById(
                "uploadDocumentTitleHelperText"
            ).style.display = "none";
            document.getElementById(
                "uploadDocumentDescriptionHelperText"
            ).style.display = "none";
            document.getElementById(
                "uploadDocumentImageHelperText"
            ).style.display = "none";
            setUploadingDocument(true);

            var formData = new FormData();
            formData.append("name", title);
            formData.append("description", description);
            formData.append("image", image[0]);
            formData.append("project", projectId);
            let response = await uploadDocument(props.auth.token, formData);
            setUploadingDocument(false);
            if (response) {
                response["image"] =
                    "data:image/png;base64," + response["image"];
                setAPIData([...APIData, response]);
                document.getElementById("uploadDocumentClose").click();
            }
        }
    };

    useEffect(() => {
        showLoader();
        async function loadProjects() {
            let projectsList = await getUserProjects(props.auth.token);
            if (projectsList) {
                setOwnedProjects(projectsList["owned_projects"]);
                setSharedProjects(projectsList["shared_projects"]);
            }
        }
        loadProjects();
    }, []);

    useEffect(() => {
        async function load() {
            if (projectId) {
                let response = await getDocuments(props.auth.token, projectId);
                if (response) {
                    for (let idx = 0; idx < response.length; idx++) {
                        response[idx]["image"] =
                            "data:image/png;base64," + response[idx]["image"];
                    }
                    let allAnnotationResponse =
                        await fetchAllAnnotationsOfAParticularProject(
                            {
                                project: projectId,
                            },
                            props.auth.token
                        );

                    if (allAnnotationResponse) {
                        setAnnotationData(allAnnotationResponse["annotations"]);
                    }
                    setAPIData(response);
                    setPageStatus("3");
                    hideLoader();
                }
            } else {
                if (ownedProjects.length > 0) {
                    history.push("/documents/" + ownedProjects[0]._id);
                } else if (sharedProjects.length > 0) {
                    history.push("/documents/" + sharedProjects[0]._id);
                } else {
                    setPageStatus("2");
                    hideLoader();
                }
            }
        }
        if (sharedProjects !== null) load();
    }, [sharedProjects, projectId]);

    // useEffect(() => {
    //     console.log("Inside use effect")

    //     async function load() {
    //         showLoader();
    //         let response =
    //     }

    //     axios.get(server+"user-document-list",
    //     {
    //         headers: {
    //             authtoken: props.auth.token,
    //         }
    //     }).then((response) => {

    //             //for now, but will be changed later
    //             for(let idx=0;idx<response.data.length;idx++){
    //                 response.data[idx]["image"] = response.data[idx]["image"];
    //             }

    //             console.log(response.data)
    //             hideLoader();
    //             setAPIData(response.data);

    //         })
    // }, [])

    // let combineData = (name,description) => {
    //     return name + " " + description
    // }

    const searchItems = (searchValue) => {
        showLoader();
        setSearchInput(searchValue);
        if (searchValue.length > 0) {
            let filterDocMap = {}; //key -> document Id , value as document
            let allDocMap = {}; //key -> document Id, value as document
            const filteredData = APIData.filter((item) => {
                allDocMap[item._id.toString()] = item;
                return (item['name'] + ' ' + item['description'])
                    .toLowerCase()
                    .includes(searchValue.toLowerCase());
            });

            for (let idx = 0; idx < filteredData.length; idx++) {
                filterDocMap[filteredData[idx]._id.toString()] =
                    filteredData[idx];
            }

            /*
                Firstly,we need to filter those annotations based on name of annotation
                then, we need to use those set of annotations and check in set of filtered documents whether document 
                with that id exists or not
            */

            const filteredAnnotationData = AnnotationData.filter((item) => {
                return item["name"]
                    .toLowerCase()
                    .includes(searchValue.toLowerCase());
            });

            for (let idx = 0; idx < filteredAnnotationData.length; idx++) {
                if (
                    !filterDocMap[filteredAnnotationData[idx]["document"]]
                ) {
                    filterDocMap[filteredAnnotationData[idx]["document"]] =
                        allDocMap[filteredAnnotationData[idx]["document"]];
                }
            }

            let finalListOfDocuments = [];
            for(let key in filterDocMap){
                finalListOfDocuments.push(filterDocMap[key]);
            }
            hideLoader();
            setFilteredResults(finalListOfDocuments);
        } else {
            hideLoader();
            setFilteredResults(APIData);
        }
    };

    const handleDocumentClick = (id) => {
        history.push(`/documents/${projectId}/${id}`);
    };

    return (
        <div>
            {loaderComponent}
            {pageStatus === "1" ? (
                <></>
            ) : pageStatus === "2" ? (
                <div style={{ paddingTop: "100px" }}>
                    <div className="d-flex justify-content-center">
                        <i className="fa-solid fa-face-frown fa-xl"></i>
                    </div>
                    <div style={{ textAlign: "center", paddingTop: "20px" }}>
                        No Projects found.
                    </div>
                </div>
            ) : (
                <div>
                    <Sidebar
                        ownedProjects={ownedProjects}
                        sharedProjects={sharedProjects}
                        project_id={projectId}
                        type="1"
                        path="/documents"
                    />
                    <Uploaddocumentmodal
                        uploadingDocument={uploadingDocument}
                        handleUploadDocument={handleUploadDocument}
                    />
                    <div>
                        <div className="cont">
                            <input
                                type="text"
                                placeholder="Search..."
                                onChange={(e) => searchItems(e.target.value)}
                            />
                            <div className="search"></div>
                        </div>
                    </div>
                    <div className="wrapper">
                        {searchInput.length >= 1
                            ? filteredResults.map((item, index) => {
                                  return (
                                      <Card
                                          id={item._id}
                                          img={item.image}
                                          title={item.name}
                                          description={item.description}
                                          key={index}
                                          handleDocumentClick={
                                              handleDocumentClick
                                          }
                                      ></Card>
                                  );
                              })
                            : APIData.map((item, index) => {
                                  return (
                                      <Card
                                          id={item._id}
                                          img={item.image}
                                          title={item.name}
                                          description={item.description}
                                          key={index}
                                          handleDocumentClick={
                                              handleDocumentClick
                                          }
                                      ></Card>
                                  );
                              })}
                    </div>
                </div>
            )}
        </div>
    );
};

let mapStateToProps = (state) => ({
    auth: state.auth,
});

export default connect(mapStateToProps)(SearchBar);
